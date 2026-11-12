from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from vnplate.config import load_config
from vnplate.data.lowlight import enhance_low_light, is_low_light
from vnplate.data.taxonomy import VEHICLE_CLASSES
from vnplate.engine.train_detector import load_detector
from vnplate.pipeline.video_pipeline import VideoVehiclePipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_detr_checkpoint(path: str | Path) -> bool:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(ckpt, dict) and "config" in ckpt


def _apply_low_light(frame: np.ndarray, mode: str, night: bool) -> np.ndarray:
    if night:
        mode = "auto" if is_low_light(frame) else "none"
    if mode in ("none", None, ""):
        return frame
    return enhance_low_light(frame, mode=mode)


def _draw(frame: np.ndarray, boxes, labels, scores, class_names) -> None:
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label = class_names[int(labels[i])] if labels is not None and int(labels[i]) < len(class_names) else "vehicle"
        score = float(scores[i]) if scores is not None else 1.0
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def _detect_detr(pipeline, frame, class_names, conf_threshold):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor, orig_size = pipeline._preprocess(frame_rgb)
    boxes, scores, labels = pipeline._detect(tensor, orig_size)
    return boxes, scores, labels, class_names


def _detect_yolo(yolo, frame, conf_threshold, imgsz, device):
    result = yolo.predict(source=frame, conf=conf_threshold, imgsz=imgsz, device=device, verbose=False)[0]
    boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
    scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.zeros((0,))
    labels = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.zeros((0,), dtype=int)
    names = result.names
    class_names = tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else tuple(names)
    return boxes, scores, labels, class_names


def _run_one(args, frame, backend, model_or_pipeline, class_names, device):
    frame = _apply_low_light(frame, args.low_light_mode, args.night)
    if backend == "detr":
        return _detect_detr(model_or_pipeline, frame, class_names, args.conf_threshold)
    return _detect_yolo(model_or_pipeline, frame, args.conf_threshold, args.imgsz, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect road users in images (DETR or YOLO).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, help="Image path or directory of images")
    parser.add_argument("--detector", required=True, help="Checkpoint: DETR (.pt with config) or YOLO (.pt)")
    parser.add_argument("--backend", default="auto", choices=["auto", "detr", "yolo"])
    parser.add_argument("--output", default=None, help="Output image/dir (default: <input>_detected)")
    parser.add_argument("--result-json", default="demo/detect_result.json")
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=None, help="DETR input size (H W)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--night", action="store_true")
    parser.add_argument("--low-light-mode", default="none", choices=["none", "auto", "clahe", "gamma", "retinex", "night"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    conf_threshold = float(args.conf_threshold if args.conf_threshold is not None else cfg.pipeline.conf_threshold)
    args.conf_threshold = conf_threshold
    device = 0 if torch.cuda.is_available() else "cpu"

    backend = args.backend
    if backend == "auto":
        backend = "detr" if _is_detr_checkpoint(args.detector) else "yolo"

    if backend == "detr":
        class_names = tuple(cfg.data.get("vehicle_classes", VEHICLE_CLASSES))
        detector = load_detector(args.detector)
        image_size = tuple(args.image_size) if args.image_size else tuple(cfg.data.image_size)
        model_or_pipeline = VideoVehiclePipeline(
            detector=detector,
            image_size=image_size,
            device=str(cfg.pipeline.device),
            conf_threshold=conf_threshold,
            class_names=class_names,
            low_light_mode=args.low_light_mode,
            night_auto=bool(args.night),
        )
    else:
        from ultralytics import YOLO

        model_or_pipeline = YOLO(args.detector)
        names = model_or_pipeline.names
        class_names = tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else tuple(names)

    in_path = Path(args.input)
    if in_path.is_dir():
        images = sorted(p for p in in_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            raise SystemExit(f"No images found in: {in_path}")
        out_dir = Path(args.output) if args.output else in_path / "detected"
        results = []
        for img in images:
            frame = cv2.imread(str(img))
            if frame is None:
                continue
            boxes, scores, labels, names_used = _run_one(args, frame, backend, model_or_pipeline, class_names, device)
            out = frame.copy()
            _draw(out, boxes, labels, scores, names_used)
            output = out_dir / f"{img.stem}_detected{img.suffix}"
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), out)
            results.append(
                {
                    "input": str(img),
                    "output": str(output),
                    "detections": [
                        {
                            "box": [float(v) for v in box],
                            "score": float(score),
                            "label": names_used[int(label)] if int(label) < len(names_used) else "vehicle",
                        }
                        for box, score, label in zip(boxes, scores, labels)
                    ],
                }
            )
        payload = {"backend": backend, "input": str(in_path), "output": str(out_dir), "count": len(results), "results": results}
    else:
        if in_path.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"Not an image: {in_path}")
        frame = cv2.imread(str(in_path))
        if frame is None:
            raise SystemExit(f"Cannot read image: {in_path}")
        boxes, scores, labels, names_used = _run_one(args, frame, backend, model_or_pipeline, class_names, device)
        out = frame.copy()
        _draw(out, boxes, labels, scores, names_used)
        output = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_detected" + in_path.suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), out)
        payload = {
            "backend": backend,
            "input": str(in_path),
            "output": str(output),
            "detections": [
                {
                    "box": [float(v) for v in box],
                    "score": float(score),
                    "label": names_used[int(label)] if int(label) < len(names_used) else "vehicle",
                }
                for box, score, label in zip(boxes, scores, labels)
            ],
        }

    Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    summary = {k: v for k, v in payload.items() if k != "results"}
    if "results" in payload:
        summary["count"] = payload["count"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
