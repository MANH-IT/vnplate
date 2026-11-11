from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

YOLO_TO_VN = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
KEEP = list(YOLO_TO_VN)


def _draw(frame, boxes, labels, scores) -> None:
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def _collect(result):
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    keep = [i for i, cls in enumerate(clss) if int(cls) in YOLO_TO_VN]
    boxes = boxes[keep]
    labels = [YOLO_TO_VN[int(clss[i])] for i in keep]
    scores = scores[keep]
    return boxes, labels, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--result-json", default="demo/yolo_detect.json")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model, e.g. yolo11n.pt / yolov8n.pt")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    model = YOLO(args.model)
    in_path = Path(args.input)
    suffix = in_path.suffix.lower()
    is_image = suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        results = model.predict(source=str(in_path), classes=KEEP, conf=args.conf_threshold, verbose=False)
        frame = cv2.imread(str(in_path))
        boxes, labels, scores = _collect(results[0])
        out = frame.copy()
        _draw(out, boxes, labels, scores)
        output = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_yolo" + suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), out)
        detections = [{"box": [float(v) for v in b], "score": float(s), "label": l} for b, l, s in zip(boxes, labels, scores)]
        payload = {"input": str(in_path), "output": str(output), "detections": detections}
    else:
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {in_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_yolo.mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frame_index = 0
        processed = 0
        start = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % args.stride != 0:
                frame_index += 1
                continue
            results = model.predict(source=frame, classes=KEEP, conf=args.conf_threshold, verbose=False)
            boxes, labels, scores = _collect(results[0])
            _draw(frame, boxes, labels, scores)
            writer.write(frame)
            processed += 1
            frame_index += 1
            if args.max_frames is not None and processed >= args.max_frames:
                break
        cap.release()
        writer.release()
        payload = {"input": str(in_path), "output": str(output), "frames_processed": processed, "fps": processed / max(time.time() - start, 1e-6)}

    Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
