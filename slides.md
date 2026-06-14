# Computer Vision & Object Detection với YOLO (v11) — Seminar lớp

> Gợi ý dùng: mỗi `## Slide` tương ứng 1 slide. Mỗi slide có **Nội dung** (để đưa vào PPT) và **Nói** (speaker notes).

---

## Slide 1 — Tiêu đề
**Nội dung**
- Computer Vision & Object Detection với YOLO (v11)
- Tên nhóm / lớp / ngày thuyết trình

**Nói**
- Hôm nay mình giới thiệu CV và Object Detection, và demo nhận diện bằng YOLO.

---

## Slide 2 — Mục tiêu buổi seminar
**Nội dung**
- Hiểu Computer Vision là gì, Object Detection là gì
- Nắm tư duy YOLO: “nhìn 1 lần” dự đoán nhiều vật thể
- Biết pipeline: dữ liệu → train → đánh giá → demo ứng dụng

**Nói**
- Mục tiêu là “hiểu + thấy được chạy thật”, không chỉ lý thuyết.

---

## Slide 3 — Computer Vision là gì?
**Nội dung**
- Computer Vision: giúp máy “hiểu” hình ảnh/video
- Input: pixel (ảnh), output: thông tin (nhãn, vị trí, hành động…)
- Ứng dụng: camera an ninh, xe tự lái, y tế, retail

**Nói**
- CV thường không chỉ “nhìn”, mà phải “ra quyết định” dựa trên ảnh.

---

## Slide 4 — 3 bài toán thị giác phổ biến
**Nội dung**
- Classification: ảnh có gì? (1 nhãn/ảnh)
- Detection: vật thể ở đâu + là gì? (bbox + class)
- Segmentation: vùng pixel nào thuộc vật thể nào? (mask)

**Nói**
- YOLO nằm ở Detection, nhưng nhiều bản có thêm Segmentation/pose.

---

## Slide 5 — Object Detection: đầu vào/đầu ra
**Nội dung**
- Input: ảnh/video
- Output (mỗi object): (x, y, w, h) + class + confidence
- Ví dụ: “person 0.86” tại bbox (…)

**Nói**
- Bbox giúp hệ thống “hành động”: đếm, cảnh báo, theo dõi (tracking)…

---

## Slide 6 — Vấn đề thực tế (khó ở đâu?)
**Nội dung**
- Ánh sáng xấu, ngược sáng
- Occlusion (che khuất), vật nhỏ
- Nhiều vật thể chồng lên nhau
- Yêu cầu real-time (FPS cao)

**Nói**
- Vì vậy model phải vừa nhanh vừa đủ chính xác.

---

## Slide 7 — YOLO là gì?
**Nội dung**
- YOLO = You Only Look Once
- 1 lần forward → dự đoán tất cả bbox + class
- Mạnh ở real-time detection

**Nói**
- So với kiểu 2-stage, YOLO thường nhanh hơn trong triển khai.

---

## Slide 8 — 2-stage vs 1-stage (trực quan)
**Nội dung**
- 2-stage (R-CNN): đề xuất vùng (proposals) → phân loại/tinh chỉnh
- 1-stage (YOLO/SSD): dự đoán trực tiếp bbox + class trên feature map
- Trade-off: tốc độ ↔ độ chính xác ↔ tài nguyên

**Nói**
- Ngày nay 1-stage cũng rất mạnh và đủ dùng cho nhiều bài toán.

---

## Slide 9 — Tiến hóa “các đời YOLO” (tổng quan)
**Nội dung**
- Ý tưởng YOLO: từ rất sớm đến nay liên tục cải tiến
- Tối ưu: backbone, neck, head, loss, augmentation, deployment
- “YOLOv11”: cách gọi tùy hệ/nhà phát hành (cần nêu rõ bạn dùng framework/weights nào)

**Nói**
- Trong seminar mình tập trung vào pipeline và kiến trúc “chuẩn YOLO”.

---

## Slide 10 — Kiến trúc tổng quát YOLO
**Nội dung**
- Backbone: trích xuất đặc trưng (features)
- Neck: kết hợp đa tỉ lệ (multi-scale, FPN/PAN…)
- Head: dự đoán bbox + class (và có thể có mask/pose)

**Nói**
- Đây là khung tư duy “Backbone–Neck–Head” dùng rất rộng rãi.

---

## Slide 11 — Multi-scale là gì? (vì sao cần)
**Nội dung**
- Vật to/vật nhỏ cần mức feature khác nhau
- Neck giúp trộn thông tin: “chi tiết” + “ngữ nghĩa”
- Kết quả: detect tốt hơn trên nhiều kích thước object

**Nói**
- Ví dụ người ở xa (nhỏ) vs xe ở gần (to).

---

## Slide 12 — Model dự đoán bbox như thế nào?
**Nội dung**
- Mạng dự đoán “tham số” để suy ra bbox
- Có thể dùng anchor-based hoặc anchor-free (tùy bản/triển khai)
- Kèm confidence (độ tin) + class probability

**Nói**
- Điểm quan trọng: đầu ra cuối cùng vẫn là bbox + class + score.

---

## Slide 13 — NMS (Non-Maximum Suppression)
**Nội dung**
- Model hay dự đoán nhiều bbox trùng nhau
- NMS: giữ bbox tốt nhất, loại bbox trùng (IoU cao)
- Tham số: IoU threshold, conf threshold

**Nói**
- NMS giúp “đầu ra sạch”, không bị 1 vật thể thành 5 bbox.

---

## Slide 14 — Dataset: COCO & định dạng label
**Nội dung**
- COCO: dataset phổ biến cho detection (nhiều class)
- Label YOLO: mỗi dòng `class x_center y_center width height` (chuẩn hóa 0..1)
- Split: train/val/test

**Nói**
- Nếu làm custom dataset, quan trọng nhất là label đúng chuẩn.

---

## Slide 15 — Chuẩn bị dữ liệu (custom)
**Nội dung**
- Thu thập ảnh phù hợp “bối cảnh” bài toán
- Gán nhãn (LabelImg/Roboflow/…)
- Kiểm tra chất lượng: sai label = model học sai

**Nói**
- Dữ liệu tốt thường quan trọng hơn “tăng model”.

---

## Slide 16 — Pipeline train (tổng quan)
**Nội dung**
- Chọn model base (nhỏ → nhanh; lớn → chính xác hơn)
- Train trên dataset → theo dõi loss/mAP
- Export (ONNX/TensorRT…) nếu triển khai

**Nói**
- Seminar thường đủ: train + chạy demo + đánh giá.

---

## Slide 17 — Tham số train quan trọng
**Nội dung**
- Epoch: số vòng lặp qua dữ liệu
- Batch size: ảnh/lần cập nhật (phụ thuộc VRAM)
- Learning rate: tốc độ học (quá cao dễ “nổ”, quá thấp học lâu)
- Image size (imgsz): ảnh lớn → chính xác hơn nhưng chậm hơn

**Nói**
- Khi demo, nói rõ bạn chọn tham số vì giới hạn máy.

---

## Slide 18 — Metric đánh giá: Precision/Recall
**Nội dung**
- Precision: dự đoán “đúng” trong số dự đoán
- Recall: tìm được “đủ” trong số object thật
- Thường có trade-off: tăng recall có thể giảm precision

**Nói**
- Tùy bài toán: an ninh có thể ưu tiên recall (đừng bỏ sót).

---

## Slide 19 — Metric đánh giá: mAP
**Nội dung**
- AP: diện tích dưới đường Precision–Recall
- mAP: trung bình AP theo class
- Thường báo mAP@0.5 và mAP@0.5:0.95

**Nói**
- Nói đơn giản: mAP càng cao thì model càng tốt (trong điều kiện so sánh giống nhau).

---

## Slide 20 — Demo: YOLO chạy real-time
**Nội dung**
- Input: webcam/video
- Output: bbox + nhãn + confidence
- Đo FPS end-to-end

**Nói**
- Demo là phần “ăn điểm”: cho lớp thấy bbox nhảy theo vật thể.

---

## Slide 21 — Code demo (ý tưởng)
**Nội dung**
- Dùng Ultralytics: load model → predict stream → vẽ kết quả
- Tham số demo: `--conf`, `--imgsz`, `--device`
- Nhấn `q` để thoát (nếu bật show)

**Nói**
- Mình đã chuẩn bị file `demo_yolo_ultralytics.py` để chạy.

---

## Slide 22 — So sánh tốc độ (FPS) theo cấu hình
**Nội dung**
- Model “nhỏ” (n): FPS cao, mAP thấp hơn
- Model “vừa/lớn” (s/m/l/x): FPS giảm, mAP tăng
- Ảnh lớn (imgsz↑): chính xác↑ nhưng chậm↓

**Nói**
- Kết luận: chọn cấu hình theo mục tiêu ứng dụng.

---

## Slide 23 — Ứng dụng thực tế
**Nội dung**
- Giao thông: người/xe, đếm xe, phát hiện vi phạm
- An ninh: phát hiện xâm nhập, cảnh báo
- Retail: đếm sản phẩm, kiểm kê, thất thoát

**Nói**
- Có thể mở rộng: tracking để theo dõi 1 đối tượng qua nhiều frame.

---

## Slide 24 — Hạn chế & thách thức
**Nội dung**
- Ánh sáng yếu / camera rung
- Occlusion, vật quá nhỏ
- Domain shift: train khác môi trường chạy thật
- Tài nguyên: CPU yếu → FPS thấp

**Nói**
- Nêu thẳng hạn chế để bài thuyết trình “thực tế”.

---

## Slide 25 — Hướng phát triển / mở rộng đề tài
**Nội dung**
- Tracking (DeepSORT/ByteTrack…) → “đếm người” ổn định hơn
- Segmentation / Pose estimation cho bài toán khó hơn
- Deployment: ONNX/TensorRT/OpenVINO cho real-time

**Nói**
- Nếu làm đồ án: chọn 1 hướng mở rộng là đủ sâu.

---

## Slide 26 — Kết luận
**Nội dung**
- CV → Detection là bài toán quan trọng
- YOLO: mạnh ở real-time, pipeline rõ ràng
- Demo cho thấy khả năng ứng dụng ngay

**Nói**
- Mời lớp đặt câu hỏi: “Vì sao nhận sai?”, “làm sao tăng mAP?”, “deploy thế nào?”…

---

## Slide 27 — Q&A
**Nội dung**
- Câu hỏi?

**Nói**
- Chuẩn bị 2–3 câu hỏi mở: dataset nào tốt? fps bao nhiêu? dùng cpu/gpu?

