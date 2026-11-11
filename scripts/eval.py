from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import torch

from vnplate.config import load_config
from vnplate.data.factory import build_detection_dataset
from vnplate.engine.evaluate import evaluate_detection
from vnplate.engine.train_detector import load_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--detector", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--conf-threshold", nargs="+", type=float, default=[0.05])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.data.split = args.split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.detector:
        detector = load_detector(args.detector, device)
        dataset = build_detection_dataset(cfg, split=args.split, synthetic_length=64)
        for conf in args.conf_threshold:
            metrics = evaluate_detection(detector, dataset, device, max_batches=args.max_batches, conf_threshold=conf)
            print(f"conf={conf:.2f} detection: {metrics}")


if __name__ == "__main__":
    main()
