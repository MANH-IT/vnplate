# VNVehicleAI — Multi-frame Vehicle Detection and Tracking

Dự án **Research** tái hiện pipeline đa khung hình cho nhận diện/tracking phương tiện giao thông trong video chất lượng thấp: **detect → track → temporal fusion** trên toàn bộ chuỗi frame.

## Kiến trúc

```
Video frame
   │
   ├─ DETR-style Detector        src/vnplate/models/detector.py
   │     ResNet/Tiny backbone + Transformer Encoder/Decoder + object queries
   │     (num_classes: car, motorbike, bus, truck)
   │
   ├─ Multi-Object Tracking       src/vnplate/tracking/
   │     KalmanBoxFilter (8D: cx,cy,w,h,vx,vy,vw,vh) + Hungarian data association
   │
   └─ Temporal Fusion             src/vnplate/fusion/temporal.py
         Weighted vote theo Track ID
```

`Track ID` (danh tính đối tượng theo thời gian) **khác** với loại phương tiện.

## Cấu trúc thư mục

```
configs/default.yaml          # toàn bộ hyperparameter
configs/low_vram.yaml         # cấu hình nhẹ cho GPU 4GB
src/vnplate/                 # package chính (alias: src/vnvehicle)
  config.py                   # config loader
  data/
    synthetic.py              # dữ liệu tổng hợp + video demo
    transforms.py             # resize, normalize
  models/
    backbone.py               # ResNet18/34/50, Tiny
    transformer.py            # encoder/decoder layers, sine pos-encoding
    detector.py               # DETR-style detector
    matcher.py                # Hungarian matcher + DETR set loss
    box_ops.py                # box conversion + GIoU
  tracking/
    kalman.py                 # Kalman filter 8D
    association.py            # IoU matrix + Hungarian
    tracker.py                # SORT-like multi-object tracker
  fusion/temporal.py          # temporal fusion
  pipeline/video_pipeline.py  # pipeline video end-to-end
  engine/
    train_detector.py
    evaluate.py               # detection / tracking metrics
scripts/                      # CLI
tests/                        # pytest
```

## Công nghệ sử dụng

- **Deep learning:** PyTorch, TorchVision
- **Detection:** DETR-style (ResNet/Tiny backbone + Transformer Encoder/Decoder)
- **Tracking:** Kalman Filter 8D + Hungarian algorithm (scipy)
- **Image/Video:** OpenCV, NumPy
- **Config:** PyYAML
- **Test:** pytest
- **Hardware:** CUDA + AMP (tùy chọn), hỗ trợ CPU

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

## Dataset phương tiện

Nếu không có dataset thật, pipeline dùng `SyntheticDetectionDataset` (nội bộ, để smoke test). Để train với dữ liệu thật, tải dataset Thai Vehicle Classification:

```bash
python scripts/download_thai_dataset.py --root data/vehicles
```

Dataset phải có layout YOLO:

```text
data/vehicles/images/train/*.jpg
data/vehicles/labels/train/*.txt
data/vehicles/images/val/*.jpg
data/vehicles/labels/val/*.txt
```

Mapping nhãn Thai → `car/motorbike/bus/truck`:

| Thai | VNPlate classes |
| --- | --- |
| car, taxi | car |
| bike | motorbike |
| bus | bus |
| pickup, truck, trailer | truck |

Đánh giá nhanh trên tập validation:

```bash
python scripts/eval.py --split val --detector checkpoints/detector_best.pt --max-batches 50
```

Sau khi tải, `build_detection_dataset` tự ưu tiên dataset YOLO này; nếu không có, fallback sang synthetic.

## Chạy pretrained baseline (nhanh)

Nếu checkpoint DETR chưa hội tụ, có thể chạy baseline pretrained để thử ảnh/video thật ngay (COCO classes: `car`, `motorbike`, `bus`, `truck`):

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

# giới hạn số frame / bỏ frame
python scripts/run_video.py --video input.mp4 --detector ... --max-frames 300 --stride 2
```

Kết quả JSON gồm `frames_processed`, `fps`, và danh sách kết quả fusion theo `track_id`.

## Đánh giá

```bash
python scripts/eval.py --split test --detector checkpoints/detector_best.pt
```

- Detection: Precision / Recall / F1 tại IoU 0.5 (greedy Hungarian matching).
- Tracking: `tracking_metrics()` trong `engine/evaluate.py` trả về IDF1 + ID switches.

## Ghi chú kỹ thuật

- **VRAM thấp (4GB):** dùng `backbone: tiny`, giảm `hidden_dim`, `num_queries`, batch size.
- **DETR cần thời gian hội tụ:** dùng pretrained backbone và AMP.
- Temporal fusion là bước triển khai bổ sung (weighted vote), không khẳng định trùng thuật toán trong paper gốc.

## License

Mã nguồn phục vụ học thuật.
