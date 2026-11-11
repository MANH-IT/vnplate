from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from vnplate.data.synthetic import write_demo_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="demo/synthetic.mp4")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--labels", nargs="+", default=["car", "motorbike"])
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out = Path(args.out)
    annotations = write_demo_video(out, labels=args.labels, frames=args.frames, image_size=(args.height, args.width), fps=args.fps)
    ann_path = out.with_suffix(".annotations.json")
    with open(ann_path, "w", encoding="utf-8") as handle:
        json.dump({"video": str(out), "labels": {str(i + 1): t for i, t in enumerate(args.labels)}, "frames": annotations}, handle, indent=2)
    print(f"wrote {out} and {ann_path}")


if __name__ == "__main__":
    main()
