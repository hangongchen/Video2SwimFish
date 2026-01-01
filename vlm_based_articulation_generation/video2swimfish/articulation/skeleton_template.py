"""Skeleton template library loader (paper Section 3.2: "a library of skeleton templates. A
template represents a valid bone segment that can be placed inside the fish body").

Reads fish_asset_pipeline/configs/skeleton_template_index.json -- the same 14
manually-constructed fish skeletons described in the paper's Section 3.5 ("Fish Skeleton
Dataset... fourteen controllable fish models... serving as fish-specific fine-tuning examples
for the actor"): a 1:1 match between the 14 template keys and the 14 .blend files under
data/fish_skeleton_dataset/<key>/<key>.blend. The index stores paths relative to the package
root; load_template_index() resolves them to absolute paths.

Plain Python, no bpy dependency -- usable from the actor/critic prompt-building code as well
as from the Blender-side articulation-editing code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import v2sf_paths as P  # noqa: E402

TEMPLATE_INDEX_PATH = P.TEMPLATE_INDEX


def _resolve(p: str | None) -> str | None:
    """Paths in skeleton_template_index.json are relative to the package root (V2SF_ROOT)."""
    if not p:
        return p
    return p if Path(p).is_absolute() else str(P.ROOT / p)


def load_template_index() -> dict:
    idx = json.loads(TEMPLATE_INDEX_PATH.read_text())
    for t in idx.get("templates", []):
        t["template_blend"] = _resolve(t.get("template_blend"))
        t["template_skeleton_json"] = _resolve(t.get("template_skeleton_json"))
    return idx


def list_templates() -> list[dict]:
    """Every template record: key, template_blend, template_skeleton_json, bone_count,
    features (size/length_height_ratio/length_thickness_ratio)."""
    return load_template_index()["templates"]


def get_template(key: str) -> dict | None:
    for t in list_templates():
        if t["key"] == key:
            return t
    return None


def template_names() -> list[str]:
    return [t["key"] for t in list_templates()]


def template_library_prompt_str() -> str:
    """Compact, human-readable description of the template library for the actor/critic
    prompt -- name, bone count, and body-shape ratios, so the VLM can pick a template whose
    body plan roughly matches the target fish."""
    lines = []
    for t in list_templates():
        f = t.get("features", {})
        lines.append(
            f"- {t['key']}: {t.get('bone_count', '?')} bones, "
            f"length/height={f.get('length_height_ratio', 0):.2f}, "
            f"length/thickness={f.get('length_thickness_ratio', 0):.2f}"
        )
    return "\n".join(lines)
