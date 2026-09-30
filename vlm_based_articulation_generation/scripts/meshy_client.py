#!/usr/bin/env python
"""Minimal Meshy Image-to-3D REST client -- image in, textured mesh out.

API details confirmed against https://docs.meshy.ai/en/api/image-to-3d and
https://docs.meshy.ai/en/api/balance before writing this (this pipeline spends real credits,
30 per meshy-6 textured task, so the request shape was verified against the docs rather than guessed):

  POST https://api.meshy.ai/openapi/v1/image-to-3d   (Authorization: Bearer <key>)
    body: image_url (public URL OR base64 data URI -- we always send a data URI, no public
    hosting needed), should_texture, texture_resolution, ai_model, target_polycount,
    target_formats, ...
    -> {"result": "<task_id>"}
  GET  https://api.meshy.ai/openapi/v1/image-to-3d/{id}   -> poll until status is a terminal
    value; SUCCEEDED carries model_urls (glb/fbx/obj/usdz/...), texture_urls, consumed_credits.
  GET  https://api.meshy.ai/openapi/v1/balance -> {"balance": N}   (read-only, free)

The API key lives ONLY in the file named by $MESHY_API_KEY_FILE (default ~/.meshy_api_key,
chmod 600) -- never pass it on a command line or commit it.
"""
from __future__ import annotations

import base64
import mimetypes
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2sf_paths as P  # noqa: E402

API_ROOT = "https://api.meshy.ai/openapi/v1"


def load_api_key() -> str:
    """Key comes from the file named by MESHY_API_KEY_FILE (default ~/.meshy_api_key)."""
    return P.meshy_key()


class MeshyClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or load_api_key()
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def balance(self) -> int:
        r = self.session.get(f"{API_ROOT}/balance", timeout=30)
        r.raise_for_status()
        return int(r.json()["balance"])

    def _image_to_data_uri(self, image_path: str) -> str:
        p = Path(image_path)
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def submit_image_to_3d(self, image_path: str, *, should_texture: bool = True,
                            texture_resolution: str = "2k", ai_model: str = "meshy-6",
                            target_polycount: int = 30000, topology: str = "triangle",
                            target_formats: list[str] | None = None,
                            enable_pbr: bool = False) -> str:
        """Submit one image-to-3d task. Returns the task id. COSTS CREDITS (30 for meshy-6 + texture)."""
        body = {
            "image_url": self._image_to_data_uri(image_path),
            "should_texture": should_texture,
            "texture_resolution": texture_resolution,
            "ai_model": ai_model,
            "target_polycount": target_polycount,
            "topology": topology,
            "target_formats": target_formats or ["glb"],
            "enable_pbr": enable_pbr,
            "should_remesh": True,
        }
        r = self.session.post(f"{API_ROOT}/image-to-3d", json=body, timeout=60)
        if r.status_code == 402:
            raise RuntimeError(f"Meshy: insufficient credits -- {r.text}")
        r.raise_for_status()
        return r.json()["result"]

    def get_task(self, task_id: str) -> dict:
        r = self.session.get(f"{API_ROOT}/image-to-3d/{task_id}", timeout=30)
        r.raise_for_status()
        return r.json()

    def poll_until_done(self, task_id: str, timeout_s: int = 900, interval_s: int = 8,
                         on_progress=None) -> dict:
        t0 = time.time()
        last_progress = -1
        while True:
            task = self.get_task(task_id)
            status = task.get("status")
            progress = task.get("progress", 0)
            if on_progress and progress != last_progress:
                on_progress(task_id, status, progress)
                last_progress = progress
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return task
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Meshy task {task_id} did not finish within {timeout_s}s "
                                    f"(last status={status}, progress={progress})")
            time.sleep(interval_s)

    def download(self, url: str, out_path: str) -> None:
        r = self.session.get(url, timeout=120)
        r.raise_for_status()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(r.content)

    def generate_and_download(self, image_path: str, out_dir: str, tag: str = "mesh",
                               on_progress=None, **submit_kwargs) -> dict:
        """End-to-end: submit -> poll -> download the glb. Returns the final task dict plus
        the local glb path under 'local_glb'. COSTS CREDITS -- call this sparingly."""
        task_id = self.submit_image_to_3d(image_path, **submit_kwargs)
        if on_progress:
            on_progress(task_id, "SUBMITTED", 0)
        task = self.poll_until_done(task_id, on_progress=on_progress)
        if task.get("status") != "SUCCEEDED":
            task["local_glb"] = None
            return task
        glb_url = task["model_urls"].get("glb")
        out_path = str(Path(out_dir) / f"{tag}.glb")
        self.download(glb_url, out_path)
        task["local_glb"] = out_path
        return task
