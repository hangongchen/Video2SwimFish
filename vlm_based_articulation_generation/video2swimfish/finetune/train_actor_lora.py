"""Fine-tune the Actor VLM (paper Section 3.2: "Actor: A_phi, a fine-tuned Qwen3-VL") via
LoRA on the SFT dataset built by build_sft_dataset.py from the 14-fish ground-truth skeleton
dataset (paper Section 3.5).

A self-contained transformers+peft LoRA trainer (NOT ms-swift): plain AdamW loop, batch size 1,
gradient accumulation, gradient checkpointing, bf16, device_map="auto". The resulting adapter
directory is a standard PEFT LoRA checkpoint, loadable exactly as
Qwen3VLClient(model_path=..., adapter_path=...).

DEFAULTS = the paper appendix (rank 8, alpha 16, 4 epochs, lr 1e-4, grad-accum 8, dropout 0.05,
target_modules="all-linear"). This reproduces the earlier `actor_lora_v1` adapter (r=8/alpha=16,
4 epochs, 300 log lines = 4 x 74 rows + 4 val lines).

The adapter that PRODUCED THE RELEASED DATASET is `actor_lora_v2`, which differs: r=16, alpha=32,
dropout 0.05, all-linear, 8 epochs, 74 train rows (train_log.jsonl: 8 epochs x 74 rows + 8 val
lines). To retrain it:

    python video2swimfish/finetune/train_actor_lora.py \
        --train outputs_sft/sft_data/rig_sft.jsonl --val outputs_sft/sft_data/rig_sft_val.jsonl \
        --output_dir checkpoints/actor_lora_v2_repro \
        --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --num_epochs 8 \
        --learning_rate 1e-4 --grad_accum_steps 8

(learning rate / grad-accum of the original v2 run were not recorded in its adapter files; the
values above are the trainer defaults and the best available assumption.)
Every run writes train_args.json next to the adapter so the exact settings are on record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import v2sf_paths as P  # noqa: E402

DEFAULT_MODEL = str(P.QWEN_MODEL_PATH)


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_example(processor, row: dict):
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in row["images"]]
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": row["prompt"]})
    messages = [{"role": "user", "content": content}]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    eos = processor.tokenizer.eos_token
    full_text = prompt_text + row["response"] + eos

    prompt_inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=images, return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[-1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels
    return full_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--num_epochs", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="all-linear",
                    help='"all-linear" (paper / v1 / v2) or a comma-separated list of module names')
    ap.add_argument("--max_steps", type=int, default=None, help="override: stop after this many optimizer steps")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[train_actor_lora] loading base model from {args.model_path} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    target_modules = args.target_modules if args.target_modules == "all-linear" \
        else [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    train_rows = load_rows(args.train)
    val_rows = load_rows(args.val) if args.val and Path(args.val).exists() else []
    print(f"[train_actor_lora] {len(train_rows)} train rows, {len(val_rows)} val rows", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")

    step = 0
    global_step = 0
    optimizer.zero_grad()
    stop = False
    for epoch in range(args.num_epochs):
        if stop:
            break
        for i, row in enumerate(train_rows):
            inputs = build_example(processor, row)
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            step += 1

            entry = {"epoch": epoch, "row": i, "fish": row.get("fish_key"), "step_in_row": row.get("step"),
                      "loss": outputs.loss.item()}
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(f"[train_actor_lora] epoch {epoch} row {i}/{len(train_rows)} "
                  f"({row.get('fish_key')} step {row.get('step')}) loss={outputs.loss.item():.4f}", flush=True)

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        if val_rows:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for row in val_rows:
                    inputs = build_example(processor, row)
                    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                    val_losses.append(model(**inputs).loss.item())
            mean_val = sum(val_losses) / len(val_losses)
            print(f"[train_actor_lora] epoch {epoch} val_loss={mean_val:.4f}", flush=True)
            log_f.write(json.dumps({"epoch": epoch, "val_loss": mean_val}) + "\n")
            log_f.flush()
            model.train()

    log_f.close()
    model.save_pretrained(str(out_dir))
    print(f"[train_actor_lora] saved LoRA adapter -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
