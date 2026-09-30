from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2sf_paths as P  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download original Qwen3-VL-32B checkpoint.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--output", default=str(P.QWEN_MODEL_PATH), help="default: $QWEN_MODEL_PATH (ROOT/models/Qwen3-VL-32B-Instruct)")
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf")
    args = parser.parse_args()

    if args.source == "hf":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub first: pip install huggingface_hub") from exc
        snapshot_download(args.model_id, local_dir=args.output, local_dir_use_symlinks=False)
    else:
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install modelscope first: pip install modelscope") from exc
        snapshot_download(args.model_id, local_dir=args.output)

    print(f"Downloaded {args.model_id} to {args.output}")


if __name__ == "__main__":
    main()
