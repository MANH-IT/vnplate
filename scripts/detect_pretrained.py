from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn_v2

COCO = {1: "person", 2: "bicycle", 3: "car", 4: "motorbike", 6: "bus", 8: "truck"}
KEEP = {3, 4, 6, 8}


def _draw(frame, boxes, labels, scores) -> None:
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{COCO[int(label)]} {score:.2f}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image or video path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--result-json", default="demo/pretrained_detect.json")
    parser.add_argument("--model", choices=["faster_rcnn", "retinanet"], default="retinanet")
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
    else:
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT").to(device).eval()

    in_path = Path(args.input)
    suffix = in_path.suffix.lower()
    is_image = suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        frame = cv2.imread(str(in_path))
        if frame is None:
            raise SystemExit(f"Cannot read image: {in_path}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        keep = (out["scores"].detach().cpu() > args.conf_threshold) & torch.tensor([int(l) in KEEP for l in out["labels"].detach().cpu()])
        boxes = out["boxes"].detach().cpu()[keep].numpy()
        labels = out["labels"].detach().cpu()[keep].numpy()
        scores = out["scores"].detach().cpu()[keep].numpy()
        out_frame = frame.copy()
        _draw(out_frame, boxes, labels, scores)
        output = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_pretrained" + suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), out_frame)
        detections = [{"box": [float(v) for v in box], "score": float(score), "label": COCO[int(label)]} for box, label, score in zip(boxes, labels, scores)]
        payload = {"input": str(in_path), "output": str(output), "detections": detections}
    else:
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {in_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_pretrained.mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frames = []
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = F.to_tensor(rgb).to(device)
            with torch.no_grad():
                out = model([tensor])[0]
            keep = (out["scores"].detach().cpu() > args.conf_threshold) & torch.tensor([int(l) in KEEP for l in out["labels"].detach().cpu()])
            boxes = out["boxes"].detach().cpu()[keep].numpy()
            labels = out["labels"].detach().cpu()[keep].numpy()
            scores = out["scores"].detach().cpu()[keep].numpy()
            _draw(frame, boxes, labels, scores)
            writer.write(frame)
            frames.append([{"box": [float(v) for v in b], "score": float(s), "label": COCO[int(l)]} for b, l, s in zip(boxes, labels, scores)])
            processed += 1
            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
        cap.release()
        writer.release()
        payload = {"input": str(in_path), "output": str(output), "frames_processed": processed, "fps": processed / max(time.time() - start, 1e-6), "frames": frames}

    Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != "frames"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
