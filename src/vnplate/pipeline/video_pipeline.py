from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from ..data.lowlight import enhance_low_light
from ..data.taxonomy import VEHICLE_CLASSES
from ..data.transforms import to_tensor
from ..fusion.temporal import TemporalFusion
from ..models.detector import DETRDetector
from ..tracking.association import cxcywh_to_xyxy, iou_matrix
from ..tracking.tracker import MultiObjectTracker


@dataclass
class FrameResult:
    frame_index: int
    detections: list[dict] = field(default_factory=list)
    tracks: list[dict] = field(default_factory=list)
    readings: list[dict] = field(default_factory=list)


class VideoVehiclePipeline:
    def __init__(
        self,
        detector: DETRDetector | None = None,
        image_size: tuple[int, int] = (384, 640),
        device: str | torch.device = "cuda",
        conf_threshold: float = 0.5,
        tracker: MultiObjectTracker | None = None,
        fusion: TemporalFusion | None = None,
        class_names: tuple[str, ...] | None = None,
        fusion_labels: bool = True,
        label_iou_threshold: float | None = None,
        low_light_mode: str = "none",
        low_light_gamma: float = 1.2,
        low_light_clip: float = 2.0,
        denoise: bool = False,
        night_auto: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.detector = detector.to(self.device).eval() if detector is not None else None
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.tracker = tracker or MultiObjectTracker()
        self.label_iou_threshold = float(label_iou_threshold) if label_iou_threshold is not None else float(self.tracker.iou_threshold)
        self.fusion = fusion or TemporalFusion()
        self.class_names = tuple(class_names) if class_names is not None else VEHICLE_CLASSES
        self.fusion_labels = fusion_labels
        self.low_light_mode = low_light_mode
        self.low_light_gamma = low_light_gamma
        self.low_light_clip = low_light_clip
        self.denoise = denoise
        self.night_auto = night_auto
        self._track_labels: dict[int, int] = {}

    def _enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        mode = self.low_light_mode
        if self.night_auto and mode in ("none", "auto"):
            from ..data.lowlight import is_low_light

            if is_low_light(frame_bgr):
                mode = "auto"
            else:
                mode = "none"
        if mode in ("none", None, ""):
            return frame_bgr
        return enhance_low_light(
            frame_bgr,
            mode=mode,
            gamma=self.low_light_gamma,
            clip_limit=self.low_light_clip,
            denoise=self.denoise,
        )

    def _preprocess(self, frame_rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        orig = frame_rgb.shape[:2]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = self._enhance(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        return to_tensor(resized).unsqueeze(0).to(self.device), orig

    def _detect(self, tensor: torch.Tensor, orig_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.detector is None:
            return np.zeros((0, 4)), np.zeros((0,)), None
        with torch.no_grad():
            result = self.detector.detect(tensor, conf_threshold=self.conf_threshold)[0]
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy() if "labels" in result else None
        if boxes.size == 0:
            return np.zeros((0, 4)), np.zeros((0,)), labels
        boxes = cxcywh_to_xyxy(boxes)
        h, w = orig_size
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        return boxes, scores, labels

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int = 0, gt_boxes: np.ndarray | None = None, gt_labels: np.ndarray | None = None) -> FrameResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor, orig_size = self._preprocess(frame_rgb)
        if gt_boxes is not None:
            boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
            scores = np.ones(len(boxes))
            labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1) if gt_labels is not None else None
        else:
            boxes, scores, labels = self._detect(tensor, orig_size)

        tracks = self.tracker.update(boxes, scores)
        result = FrameResult(frame_index=frame_index)
        result.detections = [{"box": b.tolist(), "score": float(s), "label": (self.class_names[int(l)] if labels is not None and int(l) < len(self.class_names) else None)} for b, s, l in zip(boxes, scores, labels if labels is not None else [None] * len(boxes))]

        if labels is not None and len(boxes) > 0:
            det_boxes = boxes if boxes.ndim == 2 else np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
            track_boxes = np.stack([t.bbox for t in tracks]) if tracks else np.zeros((0, 4))
            if len(track_boxes) > 0 and len(det_boxes) > 0:
                iou = iou_matrix(track_boxes, det_boxes)
                row, col = linear_sum_assignment(-iou)
                for r, c in zip(row, col):
                    if iou[r, c] < self.label_iou_threshold:
                        continue
                    track = tracks[r]
                    idx = int(c)
                    self._track_labels[track.track_id] = int(labels[idx])
                    if self.fusion_labels:
                        label_name = self.class_names[int(labels[idx])] if int(labels[idx]) < len(self.class_names) else str(labels[idx])
                        self.fusion.add(track.track_id, label_name, float(scores[idx]), frame=frame_index)
                        result.readings.append({"track_id": track.track_id, "text": label_name, "confidence": float(scores[idx])})

        for track in tracks:
            d = track.to_dict()
            if track.track_id in self._track_labels:
                d["label"] = self.class_names[self._track_labels[track.track_id]] if self._track_labels[track.track_id] < len(self.class_names) else str(self._track_labels[track.track_id])
            result.tracks.append(d)
        return result

    def process_video(self, video_path: str | Path, output_path: str | Path | None = None, max_frames: int | None = None, stride: int = 1) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frames = []
        frame_index = 0
        processed = 0
        start = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                result = self.process_frame(frame, frame_index)
                frames.append(result)
                processed += 1
                if writer is not None:
                    self._draw(frame, result)
                    writer.write(frame)
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
        cap.release()
        if writer is not None:
            writer.release()
        elapsed = time.time() - start
        return {
            "frames_processed": processed,
            "frames_total": frame_index,
            "fps": processed / max(elapsed, 1e-6),
            "final": [r.to_dict() for r in self.fusion.results(confirmed_only=False)],
            "frames": frames,
        }

    def _draw(self, frame: np.ndarray, result: FrameResult) -> None:
        for track in result.tracks:
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track['track_id']} {track.get('label', '').strip()}".strip()
            cv2.putText(frame, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        for i, reading in enumerate(result.readings):
            text = f"{reading['track_id']}: {reading['text']} ({reading['confidence']:.2f})"
            cv2.putText(frame, text, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


# Aliases
VehiclePipeline = VideoVehiclePipeline
VideoALPRPipeline = VideoVehiclePipeline
VideoDetectionPipeline = VideoVehiclePipeline
