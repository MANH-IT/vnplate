# ĐỀ CƯƠNG NGHIÊN CỨU KHOA HỌC

## Tên đề tài

**Fusion đặc trưng theo thời gian cho phát hiện – theo dõi phương tiện trong video giao thông Việt Nam chất lượng thấp**

*(Temporal Feature Fusion for Vehicle Detection and Tracking in Low-Quality Vietnamese Traffic Video)*

---

## 1. Đặt vấn đề

Giám sát giao thông bằng camera ngày càng phổ biến, nhưng dữ liệu thực tế thường **chất lượng thấp**: camera giám sát cũ, nén video mạnh, mờ do chuyển động, thiếu sáng, mưa/sương. Bên cạnh đó, giao thông Việt Nam có đặc thù **xe máy chiếm ưu thế, mật độ dày, che khuất nhiều** — khác biệt lớn so với các benchmark quốc tế (COCO, UA-DETRAC) vốn lấy ô tô làm trung tâm.

Pipeline hiện tại của nhóm (DETR + SORT + vote nhãn theo thời gian) đã chạy được nhưng chất lượng phát hiện còn thấp (F1 class-agnostic ~0.28) do: model quá nhẹ, train from scratch, vật thể nhỏ, và **fusion chỉ ở mức nhãn (post-hoc voting)** — không khai thác được thông tin đặc trưng giữa các frame.

Đề tài hướng tới hai đóng góp: (1) **fusion đặc trưng/query theo thời gian** thay cho vote nhãn, và (2) **benchmark giao thông Việt Nam chất lượng thấp** để đánh giá đúng bối cảnh thực.

---

## 2. Mục tiêu nghiên cứu

### Mục tiêu tổng quát
Nâng cao độ chính xác và độ ổn định của phát hiện–theo dõi phương tiện trong video giao thông chất lượng thấp bằng fusion đặc trưng theo thời gian, và xây dựng benchmark đánh giá phù hợp giao thông Việt Nam.

### Mục tiêu cụ thể
1. Xây dựng baseline mạnh với detector hiện đại pretrained (RT-DETR / YOLO11) + tracker (ByteTrack/OC-SORT).
2. Thiết kế module **Temporal Query Fusion (TQF)** hợp nhất đặc trưng query giữa các frame.
3. Đánh giá có kiểm chứng: **label-vote (hiện tại) vs. feature/query-level fusion (đề xuất)**.
4. Xây dựng bộ dữ liệu/benchmark video giao thông VN, có tập con chất lượng thấp (mờ, tối, mưa, nén).
5. Phân tích lỗi và đưa khuyến nghị triển khai trên thiết bị biên (edge, VRAM thấp).

---

## 3. Đối tượng và phạm vi

- **Đối tượng:** phương tiện giao thông đường bộ (ô tô, xe máy, xe tải, xe buýt, xe ba gác nếu có).
- **Dữ liệu:** video giao thông Việt Nam tự thu thập/khai thác hợp pháp + dataset công khai (Thai vehicle, UA-DETRAC) để so sánh.
- **Phạm vi:** phát hiện (2D bounding box) + theo dõi nhiều đối tượng (MOT); không bao gồm nhận dạng biển số.
- **Phần cứng mục tiêu:** GPU phổ thông (RTX 3050 4GB) → ưu tiên pretrained + đóng băng backbone + mixed precision.

---

## 4. Tổng quan tình hình nghiên cứu

### 4.1 Phát hiện đối tượng
- **DETR** (Carion et al., ECCV 2020): end-to-end, bỏ NMS; nhưng hội tụ chậm, cần lịch train dài.
- **Deformable DETR** (Zhu et al., ICLR 2021): deformable attention, multi-scale → cải thiện vật thể nhỏ.
- **DINO** (Zhang et al., ICLR 2023): denoising + anchor → DETR mạnh, hội tụ nhanh.
- **RT-DETR** (Zhao et al., CVPR 2024): DETR real-time, hiệu quả cao, có pretrained.
- **YOLO11 / YOLOv10**: họ detector CNN real-time, NMS-free (v10), mạnh về triển khai.

### 4.2 Video object detection & temporal fusion
- **FGFA** (Zhu et al., ICCV 2017), **SELSA** (Wu et al., ICCV 2019), **MEGA** (Chen et al., CVPR 2020): aggregate đặc trưng giữa các frame.
- **TransVOD** (Zhang et al., TPAMI 2022): video detection với spatial-temporal transformer.
- **Query-based video detection** (MOTR — Zeng et al., ECCV 2022): lan truyền query theo thời gian.

### 4.3 Theo dõi nhiều đối tượng (MOT)
- **ByteTrack** (Zhang et al., ECCV 2022): khai thác cả detection confidence thấp.
- **OC-SORT** (Cao et al., CVPR 2023): quan sát-centric, bền với occlusion.
- **BoT-SORT**, **StrongSORT**: kết hợp appearance (ReID).
- **SAM 2** (Ravi et al., 2024): video segmentation/tracking bằng foundation model.

### 4.4 Foundation models & label-efficient
- **DINOv2** (Oquab et al., 2023): đặc trưng thị giác mạnh cho ReID.
- **Grounding DINO / YOLO-World**: open-vocabulary detection, sinh pseudo-label.

---

## 5. Khoảng trống nghiên cứu và tính mới

1. **Fusion chủ yếu ở mức nhãn/post-hoc.** Các pipeline thực dụng thường vote nhãn sau khi detect; ít công trình đánh giá **fusion ở mức query/đặc trưng cho kiến trúc DETR** trong bối cảnh giao thông mật độ cao, chất lượng thấp.
2. **Thiếu benchmark giao thông Việt Nam/Đông Nam Á.** Các benchmark hiện có không phản ánh cảnh xe máy dày, che khuất, chất lượng thấp đặc thù.
3. **Đánh giá tách bạch đóng góp fusion vs. detector.** Cần ablation rõ: cải thiện đến từ detector mạnh hơn hay từ fusion theo thời gian.

**Đóng góp dự kiến:**
- Module **Temporal Query Fusion (TQF)** nhẹ, gắn được vào RT-DETR/DETR.
- Benchmark video giao thông VN + tập con chất lượng thấp, kèm protocol đánh giá.
- Nghiên cứu ablation có kiểm chứng label-vote vs. feature-fusion.

---

## 6. Phương pháp đề xuất

### 6.1 Kiến trúc tổng thể

```
Video frame t
   │
   ▼
[Detector pretrained: RT-DETR]  ──►  query features Q_t, boxes, scores
   │                                        │
   │                                        ▼
   │                         [Temporal Query Fusion — TQF]
   │                         ┌──────────────────────────────┐
   │                         │ Memory bank {Q_{t-K..t-1}}    │
   │                         │ Cross-frame attention        │
   │                         │ Gating + residual            │
   │                         └──────────────────────────────┘
   │                                        │
   ▼                                        ▼
[Tracker: ByteTrack/OC-SORT] ◄──── fused detections (boxes, cls, scores)
   │
   ▼
Track IDs + temporal labels
```

### 6.2 Module Temporal Query Fusion (TQF)

- **Memory bank:** lưu đặc trưng query (và vị trí) của K frame gần nhất.
- **Cross-frame attention:** query hiện tại attend sang memory để củng cố đặc trưng khi frame hiện tại mờ/che khuất.
- **Gating:** học trọng số tin cậy theo chất lượng frame (ví dụ dựa trên độ sắc nét/độ sáng) để tránh lan truyền lỗi.
- **Residual:** `Q_fused = Q_t + α · Attn(Q_t, Memory)`; α học được, khởi tạo nhỏ.
- **Huấn luyện:** đóng băng phần lớn detector, chỉ train TQF → phù hợp 4GB VRAM.
- **Suy luận:** trượt cửa sổ K frame, chi phí tăng tuyến tính nhẹ theo K (K nhỏ, ví dụ 4–8).

### 6.3 Theo dõi

- Dùng **ByteTrack** (mạnh, nhẹ) làm tracker chính; so sánh với **OC-SORT**.
- Kết hợp fused detections; đánh giá tác động của fusion lên ID switches/IDF1/HOTA.

### 6.4 Benchmark giao thông Việt Nam

- **Thu thập:** video camera giao thông (tuân thủ quy định dữ liệu cá nhân, che biển số/người nếu cần).
- **Gán nhãn:** bounding box + track ID cho các lớp: car, motorbike, bus, truck (mở rộng: xe ba gác, xe kéo).
- **Tập con chất lượng thấp:** mờ chuyển động, thiếu sáng/đêm, mưa, nén mạnh (đo bằng Laplacian variance, độ sáng, tỉ lệ nén).
- **Protocol:** chia train/val/test theo video (tránh rò rỉ frame), báo cáo mAP, MOTA, IDF1, HOTA.

---

## 7. Thực nghiệm và đánh giá

### 7.1 Dữ liệu
| Tập | Mục đích |
| --- | --- |
| Benchmark VN (đề xuất) | Đánh giá chính, bối cảnh thực |
| Thai vehicle (hiện có) | Tiền huấn luyện/so sánh |
| UA-DETRAC (nếu dùng được) | So sánh với literature |

### 7.2 Chỉ số
- Detection: **mAP@0.5:0.95, AP50, Precision/Recall/F1**.
- Tracking: **MOTA, IDF1, ID switches, HOTA**.
- Hiệu năng: FPS, VRAM, độ trễ.

### 7.3 Baseline & ablation
| Cấu hình | Mô tả |
| --- | --- |
| B0 | RT-DETR per-frame (không fusion) |
| B1 | YOLO11 per-frame |
| B2 | RT-DETR + label-vote (pipeline hiện tại) |
| **B3 (đề xuất)** | RT-DETR + **TQF** feature fusion |
| Ablation | K ∈ {2,4,8}, frozen vs. finetune, có/không gating, các tập con chất lượng thấp |

### 7.4 Câu hỏi nghiên cứu
- RQ1: Feature fusion có vượt label-vote không, và vượt bao nhiêu khi ảnh mờ/che khuất?
- RQ2: Đóng góp đến từ detector hay fusion (ablation tách bạch)?
- RQ3: Chi phí (VRAM/FPS) của TQF có chấp nhận được trên thiết bị biên?

---

## 8. Kế hoạch thực hiện (10 tuần)

| Tuần | Công việc | Đầu ra |
| --- | --- | --- |
| 1–2 | Baseline RT-DETR/YOLO11 + ByteTrack; đo trên data hiện có | Số liệu baseline, repo chạy được |
| 3 | Thiết kế TQF, prototype trên vài video | Module TQF v0 |
| 4–5 | Huấn luyện TQF (backbone frozen), tinh chỉnh | Model TQF |
| 6 | Thu thập + gán nhãn benchmark VN (đợt 1) | Dataset v0 + protocol |
| 7 | Ablation: K, gating, frozen/finetune | Bảng ablation |
| 8 | Đánh giá low-quality subsets, so sánh SOTA | Bảng kết quả chính |
| 9 | Phân tích lỗi, viết báo cáo | Bản thảo |
| 10 | Hoàn thiện, demo, nộp | Báo cáo/bài + code + dataset |

---

## 9. Tài nguyên và rủi ro

**Tài nguyên:** RTX 3050 4GB; ultralytics (RT-DETR/YOLO11) đã cài; có thể dùng Colab/Kaggle cho huấn luyện nặng.

**Rủi ro & giảm thiểu:**
| Rủi ro | Giảm thiểu |
| --- | --- |
| 4GB VRAM không đủ finetune | Đóng băng backbone, AMP, batch nhỏ, chỉ train TQF |
| Thiếu dữ liệu VN gán nhãn | Bắt đầu nhỏ, dùng pseudo-label (Grounding DINO/YOLO-World), mở rộng dần |
| Fusion không cải thiện | Ablation sớm, so sánh label-vote; có thể chuyển sang TTA/restoration-aware |
| Nộp trễ | Ưu tiên baseline + ablation cốt lõi trước, dataset mở rộng sau |

---

## 10. Sản phẩm dự kiến

1. Mã nguồn pipeline detect–track–fusion (baseline + TQF) có tài liệu.
2. Benchmark video giao thông VN + tập con chất lượng thấp + protocol đánh giá.
3. Báo cáo NCKH (và bài hội nghị nếu đủ kết quả).
4. Phân tích khuyến nghị triển khai edge.

---

## 11. Tài liệu tham khảo

1. Carion et al. *End-to-End Object Detection with Transformers (DETR)*. ECCV 2020.
2. Zhu et al. *Deformable DETR*. ICLR 2021.
3. Zhang et al. *DINO: DETR with Improved DeNoising Anchor Boxes*. ICLR 2023.
4. Zhao et al. *DETRs Beat YOLOs on Real-time Object Detection (RT-DETR)*. CVPR 2024.
5. Zhu et al. *Flow-Guided Feature Aggregation (FGFA)*. ICCV 2017.
6. Wu et al. *Sequence Level Semantics Aggregation (SELSA)*. ICCV 2019.
7. Chen et al. *MEGA: Memory Enhanced Global-Local Aggregation*. CVPR 2020.
8. Zhang et al. *TransVOD*. TPAMI 2022.
9. Zeng et al. *MOTR: End-to-End Multiple-Object Tracking with Transformer*. ECCV 2022.
10. Zhang et al. *ByteTrack*. ECCV 2022.
11. Cao et al. *OC-SORT*. CVPR 2023.
12. Ravi et al. *SAM 2: Segment Anything in Videos*. 2024.
13. Oquab et al. *DINOv2*. 2023.
14. Liu et al. *Grounding DINO*. 2023.

> Ghi chú: cần cập nhật khảo sát các công trình mới nhất tại thời điểm thực hiện để bảo đảm tính thời sự.
