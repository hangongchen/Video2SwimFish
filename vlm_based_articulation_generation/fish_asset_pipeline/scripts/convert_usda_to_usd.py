import argparse
import os

from pxr import Usd


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a USD ASCII file to .usd using the USD library.")
    parser.add_argument("--input", required=True, help="Input .usda path.")
    parser.add_argument("--output", required=True, help="Output .usd path.")
    return parser.parse_args()


def main():
    args = parse_args()
    stage = Usd.Stage.Open(args.input)
    if stage is None:
        raise RuntimeError(f"Could not open USD file: {args.input}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    stage.GetRootLayer().Export(args.output)


if __name__ == "__main__":
    main()
