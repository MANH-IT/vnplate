# VNVehicleAI — Multi-frame Road-User Detection and Tracking

Dự án **Research** tái hiện pipeline đa khung hình cho nhận diện/tracking đối tượng giao thông (người đi bộ, xe đạp, xe máy, ô tô, bus, taxi, pickup, truck, trailer) trong video chất lượng thấp: **detect → track → temporal fusion** trên toàn bộ chuỗi frame, có xử lý ban đêm (low-light) và tracking robust (ByteTrack).

## Kiến trúc

```
Video frame
   │
   ├─ Low-light Enhancer        src/vnplate/data/lowlight.py
   │     auto / clahe / gamma / retinex / night / none
   │
   ├─ DETR-style Detector        src/vnplate/models/detector.py
   │     ResNet/Tiny backbone + Transformer Encoder/Decoder + object queries
   │     (9 lớp: person, bicycle, car, motorbike, bus, taxi, pickup, truck, trailer)
   │
   ├─ Multi-Object Tracking       src/vnplate/tracking/
   │     KalmanBoxFilter (8D: cx,cy,w,h,vx,vy,vw,vh) + Hungarian
   │     + ByteTrack two-stage matching (tùy chọn)
   │
   └─ Temporal Fusion             src/vnplate/fusion/temporal.py
         Weighted vote theo Track ID
```

`Track ID` (danh tính đối tượng theo thời gian) **khác** với loại phương tiện.

## Cấu trúc thư mục

```
configs/
  default.yaml              # cấu hình đầy đủ (9 lớp)
  low_vram.yaml             # cấu hình nhẹ cho GPU 4GB
  demo_synthetic.yaml       # cấu hình demo synthetic (tiny backbone)
  vehicles_9class.yaml      # YAML cho Ultralytics YOLO baseline
  vehicles_4class.yaml       # YAML cũ (4 lớp, giữ tham khảo)
src/vnplate/                # package chính
  config.py                 # config loader
  data/
    taxonomy.py             # định nghĩa lớp đối tượng (VEHICLE_CLASSES = ROAD_USER_CLASSES)
    lowlight.py             # tăng sáng ban đêm (auto/clahe/gamma/retinex/night)
    synthetic.py            # dữ liệu tổng hợp + video demo (hỗ trợ --night)
    transforms.py           # resize, normalize
    yolo.py                 # YOLO-format dataset loader
    factory.py              # build_detection_dataset (YOLO → synthetic fallback)
  models/
    backbone.py             # ResNet18/34/50, Tiny
    transformer.py          # encoder/decoder layers, sine pos-encoding
    detector.py             # DETR-style detector
    matcher.py              # Hungarian matcher + DETR set loss
    box_ops.py              # box conversion + GIoU
  tracking/
    kalman.py               # Kalman filter 8D
    association.py          # IoU matrix + Hungarian
    tracker.py              # SORT/ByteTrack multi-object tracker
  fusion/temporal.py        # temporal fusion
  pipeline/video_pipeline.py # pipeline video end-to-end (low-light + tracking)
  engine/
    train_detector.py       # huấn luyện DETR
    evaluate.py             # detection / tracking metrics
scripts/                    # CLI
tests/                      # pytest
```

## Lớp đối tượng (9 lớp)

`person, bicycle, car, motorbike, bus, taxi, pickup, truck, trailer` (định nghĩa trong `src/vnplate/data/taxonomy.py`).

## Xử lý ban đêm / ánh sáng yếu

Module `src/vnplate/data/lowlight.py` cung cấp các chế độ tăng sáng trước detection:

- `auto`: tự động tăng gamma theo độ sáng trung bình + CLAHE
- `clahe`: cân bằng histogram cục bộ trên kênh L (LAB)
- `gamma`: gamma correction
- `retinex`: Single-Scale Retinex
- `night`: gamma + CLAHE mạnh + contrast boost
- `none`: tắt

CLI: `--night` (bật auto) hoặc `--low-light-mode clahe --low-light-clip 2.0 --denoise`.

## Tracking nâng cao

`MultiObjectTracker` hỗ trợ `byte_track=True` (two-stage matching: high-score + low-score) để giữ ID trong cảnh đông đúc/che khuất, phù hợp hướng đi của các hệ thống production (YOLOv8 + ByteTrack/BoT-SORT).

CLI: `--byte-track`.

## Công nghệ sử dụng

- **Deep learning:** PyTorch, TorchVision
- **Detection:** DETR-style (ResNet/Tiny backbone + Transformer Encoder/Decoder)
- **Tracking:** Kalman Filter 8D + Hungarian algorithm (scipy) + ByteTrack (tùy chọn)
- **Low-light:** CLAHE, Gamma, Retinex, auto night enhancement
- **Image/Video:** OpenCV, NumPy
- **Config:** PyYAML
- **Test:** pytest
- **Hardware:** CUDA + AMP (tùy chọn), hỗ trợ CPU
- **Baseline:** YOLO/RT-DETR qua ultralytics

## Cài đặt

```bash
conda env create -f environment.yml
conda activate vnplate
# PyTorch CUDA (RTX 3050 / CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

## Quickstart

```bash
# 1) Kiểm tra toàn bộ module chạy được (GPU nếu có)
python scripts/smoke_test.py

# 2) Sinh video demo tổng hợp + annotations
python scripts/make_demo.py --out demo/synthetic.mp4 --frames 90

# 3) Chạy pipeline tracking bằng GT boxes (oracle, không cần train)
python scripts/run_video.py \
  --video demo/synthetic.mp4 \
  --oracle demo/synthetic.annotations.json \
  --output demo/tracked.mp4
```

## Demo tổng hợp (synthetic)

```bash
# 1) Sinh video demo với đủ 9 loại đối tượng
python scripts/make_demo.py --out demo/synthetic9.mp4 --frames 90 \
  --labels person bicycle car motorbike bus taxi pickup truck trailer --height 256 --width 384

# 2) Train nhanh trên synthetic (9 lớp)
python scripts/train_detector.py --config configs/demo_synthetic.yaml \
  --steps 3000 --synthetic-length 16384

# 3) Chạy end-to-end: detect -> track -> temporal fusion
python scripts/run_video.py --config configs/demo_synthetic.yaml \
  --video demo/synthetic9.mp4 --detector checkpoints/detector_smoke.pt \
  --output demo/tracked9.mp4 --result-json demo/results9.json
```

Kết quả: `demo/tracked9.mp4` (video có box + track ID + nhãn) và `demo/results9.json`
(danh sách `track_id`, `text`, `confidence`, `observations`).

## Demo ban đêm (low-light + ByteTrack)

```bash
python scripts/make_demo.py --out demo/synthetic_night.mp4 --frames 60 \
  --labels person bicycle car motorbike bus truck --height 256 --width 384 --night
python scripts/run_video.py --config configs/demo_synthetic.yaml \
  --video demo/synthetic_night.mp4 --detector checkpoints/detector_smoke.pt \
  --output demo/tracked_night.mp4 --result-json demo/results_night.json \
  --night --byte-track
```

## Dataset phương tiện

Nếu không có dataset thật, pipeline dùng `SyntheticDetectionDataset` (nội bộ, để smoke test). Để train với dữ liệu thật, tải dataset Thai Vehicle Classification:

```bash
python scripts/download_thai_dataset.py --root data/vehicles
```

Dataset phải có layout YOLO:

```text
data/vehicles/<dataset>/train/images/*.jpg
data/vehicles/<dataset>/train/labels/*.txt
data/vehicles/<dataset>/valid/images/*.jpg
data/vehicles/<dataset>/valid/labels/*.txt
```

Loader cũng chấp nhận layout `images/<split>` + `labels/<split>`.

Dataset Thai gốc có 7 lớp (car, bus, taxi, motorbike, pickup, truck, trailer). Cấu hình 9 lớp nằm trong `configs/default.yaml`, `configs/low_vram.yaml` và `configs/vehicles_9class.yaml` (dùng cho baseline Ultralytics). Nếu cần gộp lớp, khai báo `data.class_map` trong config.

> Lưu ý: checkpoint cũ train với số lớp khác phải train lại; không dùng lẫn giữa các cấu hình khác nhau về số lớp.

Đánh giá nhanh trên tập validation:

```bash
python scripts/eval.py --split val --detector checkpoints/detector_best.pt --max-batches 50
```

Sau khi tải, `build_detection_dataset` tự ưu tiên dataset YOLO này; nếu không có, fallback sang synthetic.

## Chạy pretrained baseline (nhanh)

Nếu checkpoint DETR chưa hội tụ, có thể chạy baseline pretrained để thử ảnh/video thật ngay (COCO classes: `person`, `bicycle`, `car`, `motorbike`, `bus`, `truck`):

```bash
python scripts/detect_pretrained.py --input data_test/anh.jpg --output demo/test_anh_pretrained.jpg --result-json demo/test_anh_pretrained.json --conf-threshold 0.5
python scripts/detect_pretrained.py --input data_test/video.mp4 --output demo/test_video.mp4 --max-frames 30 --stride 30
```

## Huấn luyện

Với GPU 4GB, dùng cấu hình nhẹ:

```bash
python scripts/train_detector.py --config configs/low_vram.yaml
python scripts/run_video.py --config configs/low_vram.yaml --video demo/synthetic.mp4 \
  --detector checkpoints/detector_last.pt
```

Chạy thử nhanh:

```bash
python scripts/train_detector.py --steps 50
```

## Chạy trên ảnh / video

```bash
# Ảnh
python scripts/detect.py --input path/to/image.jpg --detector checkpoints/detector_best.pt --result-json demo/detect_image.json

# Ảnh ban đêm
python scripts/detect.py --input path/to/night.jpg --detector checkpoints/detector_best.pt --night --result-json demo/detect_night.json

# Video
python scripts/detect.py --input path/to/video.mp4 --detector checkpoints/detector_best.pt --output demo/detect_video.mp4 --result-json demo/detect_video.json
```

Kết quả JSON gồm các bounding box `box` (xyxy), `score`, `label` (hoặc `final` cho video).

## Chạy trên video

```bash
# với checkpoint đã train
python scripts/run_video.py \
  --video path/to/input.mp4 \
  --detector checkpoints/detector_best.pt \
  --output demo/output.mp4 \
  --result-json demo/results.json

# ban đêm + ByteTrack
python scripts/run_video.py --video input.mp4 --detector checkpoints/detector_best.pt \
  --night --byte-track --output demo/output_night.mp4

# giới hạn số frame / bỏ frame
python scripts/run_video.py --video input.mp4 --detector ... --max-frames 300 --stride 2
```

Kết quả JSON gồm `frames_processed`, `fps`, và danh sách kết quả fusion theo `track_id`.

## Đánh giá

```bash
python scripts/eval.py --split test --detector checkpoints/detector_best.pt

# thử nhiều ngưỡng confidence để chọn operating point
python scripts/eval.py --split val --detector checkpoints/detector_best.pt --conf-threshold 0.05 0.3 0.5 0.6
```

- Detection: Precision / Recall / F1 tại IoU 0.5 (greedy Hungarian matching).
- Tracking: `tracking_metrics()` trong `engine/evaluate.py` trả về IDF1 + ID switches.

## Kết quả thực nghiệm (synthetic demo)

### Cấu hình huấn luyện demo

| Thông số | Giá trị |
| --- | --- |
| GPU | NVIDIA RTX 3050 Laptop (4GB VRAM) |
| Config | `configs/demo_synthetic.yaml` (Tiny backbone, hidden_dim 128, 256×384, batch 4) |
| Steps | 1500–3000 |
| Loss | ~8.2 → ~5.3 (9 lớp) |

### Tracking + Temporal Fusion

- Tracking ổn định giữ danh tính qua che khuất (max_age frames).
- Temporal fusion (char_vote/weighted) gán nhãn nhất quán cho từng track.
- ByteTrack giữ ID tốt hơn trong cảnh đông đúc.

### Nhận xét

- **Tracking + temporal fusion hoạt động tốt**: giữ danh tính ổn định, nhãn nhất quán qua thời gian.
- **Detection cần thêm thời gian hội tụ**: DETR gốc cần hàng trăm epoch; với synthetic demo, loss giảm dần nhưng chưa đạt mAP cao.
- **Hạn chế**: precision thấp (nhiều box trùng/giả), phân loại lớp thiên lệch trên dữ liệu chưa cân bằng.
- **Hướng cải thiện**: thêm augment (flip/mosaic/color jitter), cân bằng lớp, train lâu hơn, hoặc dùng baseline YOLO11 để so sánh.

## Ghi chú kỹ thuật

- **VRAM thấp (4GB):** dùng `backbone: tiny`, giảm `hidden_dim`, `num_queries`, batch size.
- **DETR cần thời gian hội tụ:** dùng pretrained backbone và AMP.
- Temporal fusion là bước triển khai bổ sung (weighted vote), không khẳng định trùng thuật toán trong paper gốc.
- Dữ liệu ban đêm: synthetic demo mô phỏng ánh sáng yếu (giảm sáng 78%, thêm noise, tint vàng).

## License

Mã nguồn phục vụ học thuật.
