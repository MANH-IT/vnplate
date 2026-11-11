from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from vnplate.config import load_config
from vnplate.engine.train_detector import load_detector
from vnplate.fusion.temporal import TemporalFusion
from vnplate.pipeline.video_pipeline import VideoVehiclePipeline
from vnplate.tracking.tracker import MultiObjectTracker


def _load_oracle(path: str, classes: list[str] | None = None) -> dict[int, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    frames = data.get("frames", data if isinstance(data, list) else [])
    mapping: dict[int, dict] = {}
    classes = list(classes or ["car", "bus", "taxi", "motorbike", "pickup", "truck", "trailer"])
    for frame in frames:
        idx = int(frame.get("frame", 0))
        boxes = [obj["bbox"] for obj in frame.get("objects", [])]
        labels = []
        for obj in frame.get("objects", []):
            label = obj.get("label", obj.get("text", "car"))
            try:
                labels.append(classes.index(label))
            except ValueError:
                labels.append(0)
        mapping[idx] = {"boxes": boxes, "labels": labels}
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--detector", default=None)
    parser.add_argument("--oracle", default=None, help="Annotations JSON used instead of a detector (demo/debug).")
    parser.add_argument("--output", default="demo/output.mp4")
    parser.add_argument("--result-json", default="demo/results.json")
    parser.add_argument("--image-size", nargs=2, type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--night", action="store_true", help="Simulate/assume night video: enable auto low-light enhancement.")
    parser.add_argument("--low-light-mode", default="none", choices=["none", "auto", "clahe", "gamma", "retinex", "night"])
    parser.add_argument("--low-light-gamma", type=float, default=1.2)
    parser.add_argument("--low-light-clip", type=float, default=2.0)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--byte-track", action="store_true", help="Use ByteTrack-style two-stage matching.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size = tuple(args.image_size) if args.image_size else tuple(cfg.data.image_size)
    detector = load_detector(args.detector) if args.detector else None
    if detector is None and args.oracle is None:
        raise SystemExit("Provide --detector or --oracle so boxes are available.")

    tracker = MultiObjectTracker(
        max_age=int(cfg.tracking.max_age),
        min_hits=int(cfg.tracking.min_hits),
        iou_threshold=float(cfg.tracking.iou_threshold),
        use_kalman=bool(cfg.tracking.use_kalman),
        byte_track=bool(args.byte_track or cfg.tracking.get("byte_track", False)),
        high_thresh=float(cfg.tracking.get("high_thresh", 0.5)),
        low_thresh=float(cfg.tracking.get("low_thresh", 0.1)),
        new_track_thresh=float(cfg.tracking.get("new_track_thresh", 0.6)),
    )
    fusion = TemporalFusion(
        strategy=str(cfg.fusion.strategy),
        char_vote=bool(cfg.fusion.char_vote),
        min_votes=int(cfg.fusion.min_votes),
        min_confidence=float(cfg.fusion.min_confidence),
    )
    low_light_mode = "auto" if args.night else args.low_light_mode
    pipeline = VideoVehiclePipeline(
        detector=detector,
        image_size=image_size,
        device=str(cfg.pipeline.device),
        conf_threshold=float(cfg.pipeline.conf_threshold),
        tracker=tracker,
        fusion=fusion,
        class_names=tuple(cfg.data.get("vehicle_classes", ("car", "motorbike", "bus", "truck"))),
        low_light_mode=low_light_mode,
        low_light_gamma=float(args.low_light_gamma),
        low_light_clip=float(args.low_light_clip),
        denoise=bool(args.denoise),
        night_auto=bool(args.night),
    )

    if args.oracle:
        oracle = _load_oracle(args.oracle, list(cfg.data.get("vehicle_classes", [])))
        result = _run_oracle(pipeline, args.video, oracle, args.output, args.max_frames, args.stride)
    else:
        result = pipeline.process_video(args.video, output_path=args.output, max_frames=args.max_frames, stride=args.stride)

    lines = [f"frames={result['frames_processed']} fps={result['fps']:.2f}"]
    for item in result["final"]:
        lines.append(f"ID {item['track_id']}: {item['text']} conf={item['confidence']:.3f} votes={item['observations']}")
    print("\n".join(lines))
    result.pop("frames", None)
    Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def _run_oracle(pipeline, video, oracle, output, max_frames, stride):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames = []
    idx = 0
    processed = 0
    import time

    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            entry = oracle.get(idx, {"boxes": [], "labels": []})
            boxes = np.asarray(entry["boxes"], dtype=np.float64).reshape(-1, 4)
            res = pipeline.process_frame(frame, idx, gt_boxes=boxes, gt_labels=entry["labels"])
            frames.append(res)
            pipeline._draw(frame, res)
            writer.write(frame)
            processed += 1
        idx += 1
        if max_frames is not None and idx >= max_frames:
            break
    cap.release()
    writer.release()
    elapsed = time.time() - start
    return {
        "frames_processed": processed,
        "frames_total": idx,
        "fps": processed / max(elapsed, 1e-6),
        "final": [r.to_dict() for r in pipeline.fusion.results(confirmed_only=False)],
        "frames": frames,
    }


if __name__ == "__main__":
    main()
