from __future__ import annotations

import os
from pathlib import Path
from threading import Thread
from typing import Any


class Qwen3VLClient:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        debug_memory: bool = False,
        reuse_visual_inputs: bool = False,
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.debug_memory = debug_memory
        self.reuse_visual_inputs = reuse_visual_inputs
        self._model: Any | None = None
        self._processor: Any | None = None
        self._visual_input_cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self._warned_visual_reuse_disabled = False

    def load(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL runtime dependencies are missing. Install requirements.txt in a venv first."
            ) from exc

        dtype = self.torch_dtype
        if dtype == "bfloat16":
            dtype = torch.bfloat16
        elif dtype == "float16":
            dtype = torch.float16

        if self.debug_memory:
            self._print_cuda_memory("before model load")
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        # Allow disabling accelerate dispatch via env. Some older LoRA adapters
        # (saved under a slightly different Qwen3-VL module hierarchy) crash
        # during peft's post-load `dispatch_model` because accelerate's meta
        # tensors haven't been materialised yet. `QWEN3VL_DEVICE_MAP=none` skips
        # accelerate entirely: load on CPU, then .to("cuda:0").
        env_dm = os.environ.get("QWEN3VL_DEVICE_MAP", "").strip().lower()
        if env_dm in ("none", "manual", "cpu_then_cuda"):
            effective_device_map = None
        elif env_dm:
            effective_device_map = env_dm
        else:
            effective_device_map = self.device_map
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=effective_device_map,
            trust_remote_code=True,
        )
        if effective_device_map is None:
            self._model = self._model.to("cuda:0")
        if self.adapter_path:
            # peft's `_update_offload` walks safetensors prefixes and KeyErrors on
            # Qwen3-VL adapters whose saved module hierarchy has one more "model."
            # level than the currently-installed transformers exposes. The walk is
            # only meaningful for disk-offloaded weights, which we never use at
            # inference. Skip the walk so adapter loads succeed. Set
            # PEFT_SKIP_OFFLOAD_UPDATE=0 to restore the original behavior.
            if os.environ.get("PEFT_SKIP_OFFLOAD_UPDATE", "1") != "0":
                _orig = PeftModel._update_offload
                if not getattr(_orig, "_simfishlib_patched", False):
                    def _noop(self, offload_index, adapters_weights):  # noqa: ARG001
                        return offload_index
                    _noop._simfishlib_patched = True
                    PeftModel._update_offload = _noop
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()
        if self.debug_memory:
            self._print_cuda_memory("after model load")

    def generate_from_images(
        self,
        image_paths: list[str | Path],
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        show_progress: bool = False,
        progress_label: str = "generate",
    ) -> str:
        if self._model is None or self._processor is None:
            self.load()

        from PIL import Image

        content: list[dict[str, Any]] = []
        for path in image_paths:
            content.append({"type": "image", "image": Image.open(path).convert("RGB")})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [item["image"] for item in content if item["type"] == "image"]
        inputs = self._build_inputs(text, images, image_paths)
        if self.debug_memory:
            self._print_input_debug(inputs, progress_label)
        inputs = inputs.to(self._model.device)

        generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            generate_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generate_kwargs.update({"do_sample": False})

        if self.debug_memory:
            self._print_cuda_memory(f"{progress_label}: before generation")
        if show_progress:
            result = self._generate_with_progress(inputs, generate_kwargs, max_new_tokens, progress_label)
        else:
            output = self._model.generate(**inputs, **generate_kwargs)
            generated = output[:, inputs["input_ids"].shape[-1] :]
            result = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
            del output, generated
        if self.debug_memory:
            self._print_cuda_memory(f"{progress_label}: after generation")
        del inputs
        return result

    def generate_batch_from_images(
        self,
        image_paths: list[str | Path],
        prompts: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        progress_label: str = "batch",
    ) -> list[str]:
        if not prompts:
            return []
        if self._model is None or self._processor is None:
            self.load()

        from PIL import Image

        image_objects = [Image.open(path).convert("RGB") for path in image_paths]
        texts: list[str] = []
        batch_images: list[Any] = []
        for prompt in prompts:
            content: list[dict[str, Any]] = []
            for image in image_objects:
                content.append({"type": "image", "image": image.copy()})
                batch_images.append(image.copy())
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            texts.append(self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        # Decoder-only batched generation requires LEFT padding so every sequence is
        # right-aligned to a common final position; the processor defaults to right
        # padding, which corrupts generation for the padded sequences.
        tok = getattr(self._processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
        inputs = self._processor(text=texts, images=batch_images, padding=True,
                                 padding_side="left", return_tensors="pt")
        if self.debug_memory:
            self._print_input_debug(inputs, progress_label)
        inputs = inputs.to(self._model.device)

        generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            generate_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generate_kwargs.update({"do_sample": False})

        if self.debug_memory:
            self._print_cuda_memory(f"{progress_label}: before generation")
        output = self._model.generate(**inputs, **generate_kwargs)
        generated = output[:, inputs["input_ids"].shape[-1] :]
        results = self._processor.batch_decode(generated, skip_special_tokens=True)
        if self.debug_memory:
            self._print_cuda_memory(f"{progress_label}: after generation")
        del inputs, output, generated
        return results

    def _build_inputs(self, text: str, images: list[Any], image_paths: list[str | Path]) -> Any:
        if self.reuse_visual_inputs and self._get_cached_visual_inputs(image_paths) and not self._warned_visual_reuse_disabled:
            print(
                "[qwen-debug] visual-input tensor reuse is disabled for Qwen3-VL because text-only prompts can "
                "create a different number of image placeholder tokens than the cached visual features; using "
                "safe full image preprocessing.",
                flush=True,
            )
            self._warned_visual_reuse_disabled = True

        inputs = self._processor(text=[text], images=images, return_tensors="pt")
        if self.reuse_visual_inputs:
            self._cache_visual_inputs(inputs, image_paths)
        return inputs

    def _generate_with_progress(
        self,
        inputs: Any,
        generate_kwargs: dict[str, Any],
        max_new_tokens: int,
        progress_label: str,
    ) -> str:
        try:
            from tqdm.auto import tqdm
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise RuntimeError("Progress display requires tqdm and transformers.") from exc

        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_errors: list[BaseException] = []

        def run_generate() -> None:
            try:
                self._model.generate(**inputs, **generate_kwargs, streamer=streamer)
            except BaseException as exc:
                generation_errors.append(exc)
                streamer.end()

        thread = Thread(target=run_generate, daemon=True)
        thread.start()

        chunks: list[str] = []
        with tqdm(total=max_new_tokens, desc=progress_label, unit="tok") as bar:
            for chunk in streamer:
                chunks.append(chunk)
                if not chunk:
                    continue
                token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
                step = max(1, token_count)
                bar.update(min(step, max_new_tokens - bar.n))

        thread.join()
        if generation_errors:
            raise generation_errors[0]
        return "".join(chunks)

    def _get_cached_visual_inputs(self, image_paths: list[str | Path]) -> dict[str, Any] | None:
        key = tuple(str(Path(path).resolve()) for path in image_paths)
        return self._visual_input_cache.get(key)

    def _apply_cached_visual_inputs(self, inputs: Any, cached: dict[str, Any]) -> None:
        for name, value in cached.items():
            if hasattr(value, "clone"):
                inputs[name] = value.clone()
            else:
                inputs[name] = value

    def _cache_visual_inputs(self, inputs: Any, image_paths: list[str | Path]) -> None:
        """Record visual tensor shapes for repeated prompts over the same image.

        Applying cached Qwen3-VL visual tensors to a text-only processor result
        is not safe: the number of image placeholder tokens can differ across
        prompts, which causes shape mismatches in the model. We therefore keep
        the cache only for diagnostics/documentation and use full preprocessing
        for each generation.
        """
        key = tuple(str(Path(path).resolve()) for path in image_paths)
        visual_keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts")
        entry: dict[str, Any] = {}
        for name in visual_keys:
            if name not in inputs:
                continue
            value = inputs[name]
            if hasattr(value, "detach"):
                entry[name] = value.detach().cpu()
            elif hasattr(value, "clone"):
                entry[name] = value.clone()
            else:
                entry[name] = value
        if entry:
            self._visual_input_cache[key] = entry

    def _print_input_debug(self, inputs: Any, label: str) -> None:
        def shape_of(name: str) -> Any:
            if name not in inputs:
                return None
            value = inputs[name]
            return list(value.shape) if hasattr(value, "shape") else type(value).__name__

        fields = {
            "input_ids": shape_of("input_ids"),
            "attention_mask": shape_of("attention_mask"),
            "pixel_values": shape_of("pixel_values"),
            "image_grid_thw": shape_of("image_grid_thw"),
            "pixel_values_videos": shape_of("pixel_values_videos"),
            "video_grid_thw": shape_of("video_grid_thw"),
            "IMAGE_MAX_TOKEN_NUM": os.environ.get("IMAGE_MAX_TOKEN_NUM"),
        }
        print(f"[qwen-debug] {label}: input_shapes={fields}", flush=True)

    def _print_cuda_memory(self, label: str) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                print(f"[qwen-debug] {label}: cuda unavailable", flush=True)
                return
            parts = []
            for idx in range(torch.cuda.device_count()):
                parts.append(
                    {
                        "gpu": idx,
                        "allocated_gib": round(torch.cuda.memory_allocated(idx) / 1024**3, 3),
                        "reserved_gib": round(torch.cuda.memory_reserved(idx) / 1024**3, 3),
                        "max_allocated_gib": round(torch.cuda.max_memory_allocated(idx) / 1024**3, 3),
                    }
                )
            print(f"[qwen-debug] {label}: cuda_memory={parts}", flush=True)
        except Exception as exc:
            print(f"[qwen-debug] {label}: cuda memory unavailable: {exc}", flush=True)
