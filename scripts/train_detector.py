from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from vnplate.config import load_config
from vnplate.engine.train_detector import train_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None, help="Stop after N optimizer steps (useful for smoke runs).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--synthetic-length", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["detector"] = {"epochs": args.epochs}
    cfg = load_config(args.config, overrides)
    path = train_detector(cfg, max_steps=args.steps, synthetic_length=args.synthetic_length, eval_batches=args.eval_batches)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
