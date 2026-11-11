from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vnplate.config import load_config
from vnplate.engine.train_detector import load_detector
from vnplate.pipeline.video_pipeline import VEHICLE_CLASSES, VideoVehiclePipeline
from vnplate.tracking.tracker import MultiObjectTracker
from vnplate.fusion.temporal import TemporalFusion


def _draw_detections(frame: np.ndarray, boxes: np.ndarray, labels: np.ndarray | None, scores: np.ndarray | None, class_names) -> None:
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = class_names[int(labels[i])] if labels is not None and int(labels[i]) < len(class_names) else "vehicle"
        score = float(scores[i]) if scores is not None else 1.0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, help="Image or video path")
    parser.add_argument("--detector", required=True, help="Path to detector checkpoint (.pt)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--result-json", default="demo/detect_result.json")
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    class_names = tuple(cfg.data.get("vehicle_classes", VEHICLE_CLASSES))
    detector = load_detector(args.detector)
    device = str(cfg.pipeline.device)
    conf_threshold = float(args.conf_threshold if args.conf_threshold is not None else cfg.pipeline.conf_threshold)
    image_size = tuple(args.image_size) if args.image_size else tuple(cfg.data.image_size)

    tracker = MultiObjectTracker(
        max_age=int(cfg.tracking.max_age),
        min_hits=int(cfg.tracking.min_hits),
        iou_threshold=float(cfg.tracking.iou_threshold),
        use_kalman=bool(cfg.tracking.use_kalman),
    )
    fusion = TemporalFusion(
        strategy=str(cfg.fusion.strategy),
        char_vote=bool(cfg.fusion.char_vote),
        min_votes=int(cfg.fusion.min_votes),
        min_confidence=float(cfg.fusion.min_confidence),
    )
    pipeline = VideoVehiclePipeline(
        detector=detector,
        image_size=image_size,
        device=device,
        conf_threshold=conf_threshold,
        tracker=tracker,
        fusion=fusion,
        class_names=class_names,
    )

    in_path = Path(args.input)
    suffix = in_path.suffix.lower()
    is_image = suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        frame = cv2.imread(str(in_path))
        if frame is None:
            raise SystemExit(f"Cannot read image: {in_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor, orig_size = pipeline._preprocess(frame_rgb)
        boxes, scores, labels = pipeline._detect(tensor, orig_size)
        out = frame.copy()
        _draw_detections(out, boxes, labels, scores, class_names)
        output = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_detected" + suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), out)
        detections = [
            {"box": [float(v) for v in box], "score": float(score), "label": class_names[int(label)] if label is not None else "vehicle"}
            for box, score, label in zip(boxes, scores, labels if labels is not None else [None] * len(boxes))
        ]
        payload = {"input": str(in_path), "output": str(output), "detections": detections}
    else:
        output = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_detected.mp4")
        result = pipeline.process_video(in_path, output_path=output, max_frames=args.max_frames, stride=args.stride)
        payload = {"input": str(in_path), "output": str(output), "frames_processed": result["frames_processed"], "fps": result["fps"], "final": result["final"]}

    Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
