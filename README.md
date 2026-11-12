# VNVehicleAI — Nhận diện đối tượng giao thông trên ảnh

Dự án **Research** nhận diện các loại phương tiện giao thông trên **ảnh** (ưu tiên), có hỗ trợ thêm pipeline video tracking. Hệ thống cung cấp **2 backend**:

- **YOLO11 (ultralytics)** — khuyến nghị, hội tụ nhanh, chất lượng cao.
- **DETR-style (PyTorch tự viết)** — kiến trúc nghiên cứu: ResNet + Transformer Encoder/Decoder + object queries.

Kèm module **tăng sáng ban đêm (low-light)** và (tùy chọn) **tracking + temporal fusion** cho video.

## Tính năng

- Nhận diện trên **1 ảnh** hoặc **cả thư mục ảnh**, xuất ảnh có bounding box + JSON kết quả.
- Tự nhận diện backend từ checkpoint (`--backend auto`): DETR hay YOLO.
- Tăng sáng trước khi detect: `auto / clahe / gamma / retinex / night / none`.
- (Tùy chọn) Video tracking: Kalman 8D + Hungarian, hoặc ByteTrack two-stage.
- (Tùy chọn) Temporal fusion theo Track ID cho video.

## Hai backend

| | YOLO11 | DETR-style |
| --- | --- | --- |
| Cài đặt | `ultralytics` | PyTorch thuần (trong repo) |
| Số lớp | **7** (Thai) | **9** |
| Tên lớp | car, bus, taxi, bike, pickup, truck, trailer | person, bicycle, car, motorbike, bus, taxi, pickup, truck, trailer |
| Hội tụ | Nhanh (vài epoch) | Chậm (cần hàng trăm epoch) |
| Trạng thái | **Khuyến nghị** | Tham khảo/nghiên cứu |

> Lưu ý: DETR 9 lớp dùng `data.class_map` để ánh xạ 7 lớp Thai vào chỉ số 2–8. Hai lớp `person`/`bicycle` **chưa có dữ liệu huấn luyện**.

## Cấu trúc thư mục

```
configs/
  thai_yolo.yaml          # YOLO 7 lớp (Thai) — dùng cho nhận diện ảnh
  thai_9class.yaml        # DETR 9 lớp trên dữ liệu Thai (class_map 7->9)
  default.yaml            # DETR cấu hình đầy đủ (9 lớp)
  low_vram.yaml           # DETR nhẹ cho GPU 4GB
  demo_synthetic.yaml     # DETR demo synthetic (tiny backbone)
  vehicles_9class.yaml    # YAML YOLO cũ (tham khảo)
  vehicles_4class.yaml    # YAML YOLO cũ 4 lớp (tham khảo)
src/vnplate/
  config.py               # config loader
  data/
    taxonomy.py           # định nghĩa lớp đối tượng (VEHICLE_CLASSES)
    lowlight.py           # tăng sáng (auto/clahe/gamma/retinex/night)
    yolo.py               # YOLO-format dataset loader (hỗ trợ class_map)
    factory.py            # build_detection_dataset (YOLO -> synthetic fallback)
    synthetic.py          # dữ liệu tổng hợp + video demo
    transforms.py         # resize, normalize, box conversion
  models/                 # DETR: backbone, transformer, detector, matcher, box_ops
  tracking/               # kalman, association, tracker (SORT/ByteTrack)
  fusion/temporal.py      # temporal fusion
  pipeline/video_pipeline.py  # pipeline video (low-light + tracking)
  engine/
    train_detector.py     # huấn luyện DETR
    evaluate.py           # detection / tracking metrics
scripts/
  detect.py               # nhận diện ảnh (DETR hoặc YOLO)
  train_yolo_baseline.py  # huấn luyện YOLO11
  train_detector.py       # huấn luyện DETR
  eval.py                 # đánh giá DETR
  run_video.py            # pipeline video (tùy chọn)
  make_demo.py            # sinh video synthetic
  smoke_test.py           # kiểm tra module
  download_thai_dataset.py
tests/                    # pytest
```

## Cài đặt

```bash
conda env create -f environment.yml
conda activate vnplate

# PyTorch CUDA (RTX 3050 / CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics          # cho backend YOLO
pip install -e .
```

## Dataset (Thai Vehicle — 7 lớp)

```bash
python scripts/download_thai_dataset.py --root data/vehicles
```

Layout YOLO:

```text
data/vehicles/<dataset>/train/images/*.jpg
data/vehicles/<dataset>/train/labels/*.txt
data/vehicles/<dataset>/valid/images/*.jpg
data/vehicles/<dataset>/valid/labels/*.txt
```

Phân bố lớp (train, ~24k box): car 8709, pickup 5061, bus 3911, bike 2217, truck 1984, taxi 1257, trailer 914.

`configs/thai_yolo.yaml` khai báo đúng 7 lớp cho backend YOLO. `configs/thai_9class.yaml` dùng `data.class_map: {0:2, 1:4, 2:5, 3:3, 4:6, 5:7, 6:8}` cho DETR 9 lớp.

## Quickstart — nhận diện ảnh

### Với YOLO (khuyến nghị)

```bash
# 1 ảnh
python scripts/detect.py --config configs/thai_yolo.yaml \
  --input data_test/anh.jpg \
  --detector runs/detect/runs/baseline/yolo11n_thai/weights/best.pt \
  --output demo/anh_detected.jpg --conf-threshold 0.25

# cả thư mục
python scripts/detect.py --config configs/thai_yolo.yaml \
  --input data_test \
  --detector runs/detect/runs/baseline/yolo11n_thai/weights/best.pt \
  --output demo/yolo_detected --conf-threshold 0.2
```

### Với DETR

```bash
python scripts/detect.py --config configs/thai_9class.yaml \
  --input data_test/anh.jpg \
  --detector checkpoints/detector_best.pt \
  --result-json demo/detr_result.json
```

Backend được tự phát hiện; có thể ép bằng `--backend yolo` hoặc `--backend detr`.

### Ảnh tối / ban đêm

```bash
python scripts/detect.py --config configs/thai_yolo.yaml \
  --input data_test/anh_4.jpg \
  --detector runs/detect/runs/baseline/yolo11n_thai/weights/best.pt \
  --low-light-mode clahe      # hoặc --night
```

Kết quả JSON: danh sách `box` (xyxy), `score`, `label`.

## Demo trên ảnh thực tế

Ảnh đường phố thực tế trong `data_test/`, nhận diện bằng YOLO pretrained COCO (`yolo11n.pt`):

| Ảnh gốc | Sau nhận diện |
| --- | --- |
| ![Ảnh gốc](assets/test_anh_original.jpg) | ![Sau nhận diện](assets/test_anh_detected.jpg) |
| ![Ảnh gốc](assets/test_anh3_original.jpg) | ![Sau nhận diện](assets/test_anh3_detected.jpg) |

Ảnh trực tiếp:

![Demo 1 - sau nhận diện](assets/test_anh_detected.jpg)

![Demo 2 - sau nhận diện](assets/test_anh3_detected.jpg)

Kết quả: **23** đối tượng ở ảnh 1 (car 14, bus 5, person 3, motorcycle 1) và **14** ở ảnh 2 (car 9, bus 2, person 2, truck 1).

```bash
python scripts/detect.py --config configs/thai_yolo.yaml \
  --input data_test/anh.jpg --detector yolo11n.pt \
  --output assets/test_anh_detected.jpg --conf-threshold 0.35
```

> Lưu ý: model huấn luyện trên dữ liệu Thai hiện có **domain gap** với ảnh ngoài miền (confidence thấp), nên demo trên dùng model COCO pretrained. Để model tự huấn luyện bắt tốt ảnh thực tế, cần bổ sung ảnh cùng miền vào tập train.

## Huấn luyện

### YOLO11 (khuyến nghị)

```bash
python scripts/train_yolo_baseline.py \
  --model yolo11n.pt --data configs/thai_yolo.yaml \
  --epochs 50 --imgsz 640 --batch 4 --device 0 --workers 4 \
  --name yolo11n_thai
```

Trọng số: `runs/detect/runs/baseline/yolo11n_thai/weights/best.pt`. Script tự chạy `val` và in mAP cuối.

**Resume** nếu training bị ngắt (dùng `last.pt`, đọc lại args đã lưu):

```bash
python scripts/train_yolo_baseline.py --resume \
  --model runs/detect/runs/baseline/yolo11n_thai/weights/last.pt
```

> Trên RTX 3050 4GB: ~2.5 phút/epoch, 50 epoch ~2.5 giờ. Có thể giảm `--epochs`, `--imgsz` để nhanh hơn.

### DETR

```bash
python scripts/train_detector.py --config configs/thai_9class.yaml
# chạy thử nhanh
python scripts/train_detector.py --config configs/thai_9class.yaml --steps 50
```

## Đánh giá

```bash
# DETR trên tập validation, thử nhiều ngưỡng confidence
python scripts/eval.py --config configs/thai_9class.yaml --split val \
  --detector checkpoints/detector_best.pt --max-batches 50 \
  --conf-threshold 0.3 0.5 0.6
```

YOLO tự đánh giá sau khi train (`FINAL mAP50-95=... mAP50=...`).

## Kết quả thực nghiệm

### YOLO11n trên Thai (val, 3 epoch)

| Metric | Giá trị |
| --- | --- |
| mAP50 | **0.652** |
| mAP50-95 | **0.421** |

| Class | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| car | 0.566 | 0.701 | 0.664 | 0.416 |
| bus | 0.708 | 0.793 | 0.817 | 0.602 |
| taxi | 0.657 | 0.716 | 0.734 | 0.462 |
| bike | 0.670 | 0.734 | 0.678 | 0.319 |
| pickup | 0.382 | 0.570 | 0.456 | 0.300 |
| truck | 0.686 | 0.522 | 0.645 | 0.435 |
| trailer | 0.360 | 0.694 | 0.564 | 0.414 |

Chỉ sau **3 epoch** YOLO đã vượt xa DETR (F1 ~0.35 sau 14 epoch). Train đủ 50 epoch cho kết quả tốt hơn.

### DETR (Thai, 14 epoch)

| Conf | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 0.30 | 0.131 | 0.551 | 0.212 |
| 0.50 | 0.231 | 0.406 | 0.295 |
| 0.60 | 0.329 | 0.377 | 0.351 |

### Nhận xét

- **YOLO vượt trội** về tốc độ hội tụ và độ chính xác trên dữ liệu thật.
- **DETR** cần hàng trăm epoch; hiện chưa hội tụ.
- **Domain gap**: ảnh ngoài miền Thai (tối, cảnh khác) cho confidence thấp — cần bổ sung dữ liệu tương tự vào train để cải thiện.

## (Tùy chọn) Video + tracking

```bash
python scripts/run_video.py --config configs/thai_9class.yaml \
  --video path/to/input.mp4 --detector checkpoints/detector_best.pt \
  --output demo/output.mp4 --result-json demo/results.json

# ban đêm + ByteTrack
python scripts/run_video.py --config configs/thai_9class.yaml \
  --video input.mp4 --detector checkpoints/detector_best.pt \
  --night --byte-track --output demo/output_night.mp4
```

Pipeline video: `detect → track → temporal fusion`. `run_video.py` dùng backend DETR.

## Kiểm thử

```bash
python scripts/smoke_test.py
pytest -q
```

## Ghi chú kỹ thuật

- **VRAM thấp (4GB):** giảm `--batch`, `--imgsz`; DETR dùng `backbone: tiny`, giảm `hidden_dim`/`num_queries`.
- **DETR cần thời gian hội tụ:** dùng pretrained backbone + AMP.
- **Số lớp phải khớp checkpoint:** checkpoint 7 lớp ≠ 9 lớp; không dùng lẫn.
- **`configs/vehicles_9class.yaml`** là YAML cũ (tên lớp không khớp dataset Thai) — dùng `configs/thai_yolo.yaml`.

## License

Mã nguồn phục vụ học thuật.
