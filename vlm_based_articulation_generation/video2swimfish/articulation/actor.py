"""Actor VLM (paper Section 3.2, Eq. 3: a^(k) = A_phi(M, K^(k))).

"Fine-tuned Qwen3-VL, or the pretrained model if no fine-tuned checkpoint is available."

The actor is the base Qwen3-VL-32B-Instruct plus a LoRA adapter (`checkpoint_path`; the
shipped one is `actor_lora_v2`, see README "Actor LoRA provenance"). The adapter is loaded ONTO
the shared model by Qwen3VLClient(adapter_path=...); `_set_adapter_enabled` switches it on for
the actor and off for the critic, so one 32B instance serves both. If `checkpoint_path` is
None / missing the actor silently degrades to the pretrained model (`is_finetuned=False`),
and if the VLM cannot be loaded at all it falls back to a MOCK actor -- always check
`actor_meta.source == "qwen3vl"` in articulation_construction_log.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # package root: v2sf_paths + simfishlib/

ACTOR_PROMPT_TEMPLATE = """You are an expert in fish skeletal anatomy. You are constructing
an internal skeleton for a fish mesh.

In the rendered views, the fish body (skin) is shown as a faint, translucent gray silhouette
so you can see through it. The bones are the small, bright glowing orange shapes -- everything
orange/glowing is a bone; everything faint gray is the body surface, not a bone.

Current articulation: {current_articulation_json}
Critic feedback from last iteration: {critic_feedback}

Available skeleton templates: {template_library}

Rules:
1. All bones must be placed INSIDE the fish body, not protruding outside
2. The skeleton must cover continuously from head to caudal fin
3. Bones must follow the fish's longitudinal body axis

Looking at the rendered views of the current articulation, propose
ONE modification (add/remove/reposition/resize a bone). Joints are not part of
this task -- every joint is driven by a fixed physics preset applied later in the
pipeline, so do not propose joint parameters.

Respond ONLY in JSON format:
{{
  "action": "add" | "remove" | "reposition" | "resize",
  "bone_id": <int or null for add>,
  "template": <template_name>,
  "position": [x, y, z],
  "size": [l, w, h],
  "reasoning": <one sentence explaining why>
}}"""


def _extract_json(text: str) -> dict | None:
    """Pretrained (non-fine-tuned) Qwen3-VL is not guaranteed to emit strict JSON, so scan
    for the first balanced {...} block rather than assuming the whole response is JSON."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _set_adapter_enabled(client, enabled: bool) -> None:
    """Toggle the shared client's PEFT LoRA adapter layers on/off in place, so ONE loaded
    32B model instance can serve both the fine-tuned actor (adapter ON) and the pretrained
    critic (adapter OFF, paper Sec 3.2: critic has no fine-tuning). `client._model` is a
    PeftModel only when Qwen3VLClient was constructed with adapter_path set; a plain
    (non-adapted) model has neither method, so this is a harmless no-op in that case."""
    model = getattr(client, "_model", None)
    if model is None:
        return
    method = "enable_adapter_layers" if enabled else "disable_adapter_layers"
    if hasattr(model, method):
        getattr(model, method)()


class Actor:
    def __init__(self, client=None, checkpoint_path: str | None = None):
        """`client`: a loaded Qwen3VLClient (shared with Critic -- one 32B model instance,
        adapter layers toggled per call via _set_adapter_enabled -- to avoid holding two
        copies of a 32B model in GPU memory at once, a real constraint hit earlier this
        session running two Qwen3-VL instances concurrently)."""
        self.client = client
        self.checkpoint_path = checkpoint_path
        self.is_finetuned = checkpoint_path is not None and Path(str(checkpoint_path)).exists()

    def propose(self, image_paths: list[str], current_articulation_json: dict,
                critic_feedback, template_library: str, mesh_bbox: dict | None = None,
                temperature: float = 0.0) -> dict:
        """Returns the actor's action dict, plus a `_meta` key recording whether the real
        VLM produced it or the mock fallback fired (for the iteration log)."""
        prompt = ACTOR_PROMPT_TEMPLATE.format(
            current_articulation_json=json.dumps(current_articulation_json),
            critic_feedback=json.dumps(critic_feedback) if critic_feedback else "(none, this is the first iteration)",
            template_library=template_library,
        )

        if self.client is not None:
            _set_adapter_enabled(self.client, True)
            try:
                raw = self.client.generate_from_images(
                    image_paths=image_paths, prompt=prompt, max_new_tokens=1024, temperature=temperature,
                )
                parsed = _extract_json(raw)
                if parsed is not None and "action" in parsed:
                    parsed["_meta"] = {"source": "qwen3vl", "finetuned": self.is_finetuned, "raw": raw, "temperature": temperature}
                    return parsed
                # one retry with a stricter reminder -- pretrained models often wrap JSON in
                # prose despite "Respond ONLY in JSON format"
                raw2 = self.client.generate_from_images(
                    image_paths=image_paths,
                    prompt=prompt + "\n\nReminder: output ONLY the JSON object, no other text.",
                    max_new_tokens=1024, temperature=temperature,
                )
                parsed2 = _extract_json(raw2)
                if parsed2 is not None and "action" in parsed2:
                    parsed2["_meta"] = {"source": "qwen3vl_retry", "finetuned": self.is_finetuned, "raw": raw2}
                    return parsed2
                fallback = self._mock_action(current_articulation_json, mesh_bbox)
                fallback["_meta"] = {"source": "mock_parse_failed", "finetuned": self.is_finetuned,
                                       "raw": raw, "raw_retry": raw2}
                return fallback
            except Exception as e:  # noqa: BLE001 -- any VLM failure must degrade, never crash the loop
                fallback = self._mock_action(current_articulation_json, mesh_bbox)
                fallback["_meta"] = {"source": "mock_exception", "finetuned": self.is_finetuned, "error": str(e)}
                return fallback

        fallback = self._mock_action(current_articulation_json, mesh_bbox)
        fallback["_meta"] = {"source": "mock_no_client", "finetuned": self.is_finetuned}
        return fallback

    def _mock_action(self, current_articulation_json: dict, mesh_bbox: dict | None) -> dict:
        """Placeholder used only when Qwen3-VL is unavailable or fails to parse (explicitly
        sanctioned: "mock the VLM responses with a placeholder that still runs the full
        pipeline structure"). Positions are computed from the REAL mesh bounding box passed
        in by the orchestrator (never a fixed constant) so this still respects "do not
        hardcode any bone positions": it builds a head-to-tail chain, evenly spaced along the
        actual measured length of THIS fish, one bone added per call."""
        from skeleton_template import list_templates

        bones = current_articulation_json.get("bones", [])
        templates = list_templates()
        n_target = max((t.get("bone_count", 6) for t in templates), default=6)
        template = min(templates, key=lambda t: abs(t.get("bone_count", 6) - n_target))

        if mesh_bbox is None:
            mesh_bbox = {"min": [-0.15, -0.03, -0.03], "max": [0.15, 0.03, 0.03]}
        xmin, ymin, zmin = mesh_bbox["min"]
        xmax, ymax, zmax = mesh_bbox["max"]
        length = xmax - xmin
        n_bones = template.get("bone_count", 6)

        if len(bones) < n_bones:
            idx = len(bones)
            frac = (idx + 0.5) / n_bones
            x = xmin + frac * length
            seg_len = length / n_bones
            return {
                "action": "add",
                "bone_id": None,
                "template": template["key"],
                "position": [x, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0],
                "size": [seg_len * 0.9, (ymax - ymin) * 0.6, (zmax - zmin) * 0.6],
                "reasoning": "mock actor: extending the head-to-tail chain by one segment "
                             "spaced along the fish's measured bounding-box length",
            }
        last = bones[-1]
        return {
            "action": "resize",
            "bone_id": last["bone_id"],
            "template": last.get("template", template["key"]),
            "position": last["position"],
            "size": [s * 0.95 for s in last["size"]],
            "reasoning": "mock actor: chain is complete, minor tail-bone taper adjustment",
        }
