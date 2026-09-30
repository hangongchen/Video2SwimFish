"""Critic VLM (paper Section 3.2, Eq. 5: (r^(k), f^(k)) = C_psi(M, K^(k+1))).

"Pretrained Qwen3-VL, no fine-tuning" -- unlike the actor, the paper does NOT expect the
critic to ever be fine-tuned, so this class has no checkpoint_path at all; it is always the
base Qwen3-VL-32B-Instruct.
"""
from __future__ import annotations

import json

CRITIC_PROMPT_TEMPLATE = """You are a biology expert evaluating a fish skeleton placement.
Evaluate the proposed skeleton inside the fish mesh shown in
the rendered views.

In these renders, the fish body (skin) is shown as a faint, translucent gray silhouette so
you can see through it. The bones are the small, bright glowing orange shapes -- everything
orange/glowing is a bone; everything faint gray is the body surface, not a bone. The three
images are: (1) a side/profile view, (2) a head-on view looking down the body axis -- in this
view a correctly aligned chain of bones stacks up into ONE compact orange blob or cross in
the middle of the body outline, which is CORRECT, not a defect -- and (3) a top-down view.
Judge coverage and alignment from views (1) and (3); use view (2) only to check that the
blob sits in the centre of the cross-section.

Proposed articulation: {proposed_articulation_json}

Measured facts about this articulation (computed from the 3D geometry -- trust these over
your own reading of the images when they disagree):
{measured_facts}

Evaluation criteria:
1. No bones extend outside the fish body surface
2. Continuous coverage from head to caudal fin
3. MIDLINE ALIGNMENT -- the most important criterion. The vertebral column runs through the
   CENTER of the body cross-section. In EVERY view the bone chain must sit in the middle of
   the gray silhouette with roughly equal skin margin on both sides of it. A chain that runs
   along the belly, along the back, or along one flank is WRONG even if it is inside the body
   and covers the length. Use the measured offsets: |offset| 0.0 = on the midline, 1.0 = at
   the skin. Any bone with |offset| > 0.35 on either axis, or a mean |offset| > 0.2, means
   the skeleton is NOT on the midline: score at most 2 and accepted=false, and say which
   direction it is displaced (dorsal/ventral/left/right).
4. Bone shapes and sizes are biologically plausible for this fish morphology
5. Appropriate number of bones for this fish morphology

Note: joints are not part of this task -- every joint is driven by a fixed physics
preset applied later in the pipeline, so do not evaluate joint parameters.

Respond ONLY in JSON format:
{{
  "score": <int 1-5>,
  "accepted": <bool>,
  "feedback": {{
    "interpenetration": <str or null>,
    "coverage": <str or null>,
    "alignment": <str or null>,
    "bone_shape": <str or null>,
    "other": <str or null>
  }}
}}
Score guide: 1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent
accepted=True only if score >= 4"""


def _extract_json(text: str) -> dict | None:
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
    """See actor.py's copy of this helper: toggles the shared client's PEFT LoRA adapter
    on/off so the critic runs the UNTOUCHED pretrained model even when the actor's adapter is
    loaded into the same shared model instance. No-op if the client isn't PEFT-wrapped."""
    model = getattr(client, "_model", None)
    if model is None:
        return
    method = "enable_adapter_layers" if enabled else "disable_adapter_layers"
    if hasattr(model, method):
        getattr(model, method)()


def _measured_facts_text(verification: list[dict] | None, midline_offsets: list[dict] | None) -> str:
    lines = []
    for v in verification or []:
        if not v:
            continue
        lines.append(f"- containment of the bone just modified ({v.get('bone_name')}): "
                     f"{v.get('vertices_outside_after', '?')} of {v.get('vertices_total_after', '?')} "
                     f"vertices outside the skin after correction")
    if midline_offsets:
        offs = [(o["bone"], o["offset_y_frac"], o["offset_z_frac"]) for o in midline_offsets]
        mean_y = sum(abs(o[1]) for o in offs) / len(offs)
        mean_z = sum(abs(o[2]) for o in offs) / len(offs)
        worst = max(offs, key=lambda o: max(abs(o[1]), abs(o[2])))
        lines.append(f"- midline offset of each bone (fraction of local body half-extent; y and z axes): "
                     + ", ".join(f"{b}: y={oy:+.2f} z={oz:+.2f}" for b, oy, oz in offs))
        lines.append(f"- mean |offset|: y={mean_y:.2f} z={mean_z:.2f}; worst bone {worst[0]} "
                     f"(y={worst[1]:+.2f}, z={worst[2]:+.2f})")
    return "\n".join(lines) if lines else "- (none available this iteration)"


class Critic:
    def __init__(self, client=None):
        self.client = client

    def evaluate(self, image_paths: list[str], proposed_articulation_json: dict,
                 geometric_verification_report: list[dict] | None = None,
                 midline_offsets: list[dict] | None = None) -> dict:
        """Returns {"score", "accepted", "feedback", "_meta"}. The hard geometric-verification
        result is passed in for context/logging only -- it is already guaranteed true (bones
        were forcibly shrunk to fit before the critic ever sees them), so the critic's job is
        everything ELSE the geometric check cannot see: coverage, alignment, joint plausibility,
        bone count appropriateness."""
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            proposed_articulation_json=json.dumps(proposed_articulation_json),
            measured_facts=_measured_facts_text(geometric_verification_report, midline_offsets),
        )

        if self.client is not None:
            _set_adapter_enabled(self.client, False)
            try:
                raw = self.client.generate_from_images(
                    image_paths=image_paths, prompt=prompt, max_new_tokens=1024, temperature=0.0,
                )
                parsed = _extract_json(raw)
                if parsed is not None and "score" in parsed:
                    return self._normalize(parsed, source="qwen3vl", raw=raw)
                raw2 = self.client.generate_from_images(
                    image_paths=image_paths,
                    prompt=prompt + "\n\nReminder: output ONLY the JSON object, no other text.",
                    max_new_tokens=1024, temperature=0.0,
                )
                parsed2 = _extract_json(raw2)
                if parsed2 is not None and "score" in parsed2:
                    return self._normalize(parsed2, source="qwen3vl_retry", raw=raw2)
                return self._mock_evaluate(geometric_verification_report, source="mock_parse_failed",
                                             raw=raw, raw_retry=raw2)
            except Exception as e:  # noqa: BLE001 -- a VLM failure must degrade, never crash the loop
                return self._mock_evaluate(geometric_verification_report, source="mock_exception", error=str(e))

        return self._mock_evaluate(geometric_verification_report, source="mock_no_client")

    def _normalize(self, parsed: dict, **meta) -> dict:
        score = int(parsed.get("score", 1))
        parsed["score"] = score
        parsed["accepted"] = bool(parsed.get("accepted", score >= 4))
        parsed.setdefault("feedback", {})
        parsed["_meta"] = meta
        return parsed

    def _mock_evaluate(self, geometric_verification_report, **meta) -> dict:
        """Placeholder used only when Qwen3-VL is unavailable or fails to parse (explicitly
        sanctioned by the user). Bases its score on the one thing this pipeline can check
        without a VLM at all: the hard geometric-verification result (bones fully inside the
        skin or not) -- everything the criteria list asks that ISN'T visual judgement."""
        reports = geometric_verification_report or []
        all_contained = all(r.get("fully_contained", False) for r in reports) if reports else False
        score = 3 if all_contained else 2
        return {
            "score": score,
            "accepted": False,  # mock never auto-accepts; real VLM judgement is required to converge
            "feedback": {
                "interpenetration": "all bones geometrically verified inside the skin"
                if all_contained else "some bones failed geometric containment",
                "coverage": None,
                "alignment": None,
                "bone_shape": None,
                "other": "mock critic: Qwen3-VL unavailable, score derived only from the "
                          "geometric-verification hard constraint, not visual judgement",
            },
            "_meta": meta,
        }
