# Research Log

## 2026-10-03
- Làm: cài môi trường, chạy pose trên ảnh mẫu và webcam
- Quan sát: 
    Python     : 3.11.9
    OpenCV     : 5.0.0
    NumPy      : 2.4.6
    PyTorch    : 2.14.1+cpu
    Ultralytics: 8.4.171
    GPU CUDA   : False
- Câu hỏi mở:
- Tiếp theo: Milestone 1.1

## 2026-10-03 — Milestone 1.1

- Mục tiêu: Đọc output YOLO11n-pose trên một ảnh tĩnh — khung bao, 17 keypoint COCO, confidence từng khớp. Chưa đo té ngã.
- Cấu hình:
  - Script: `research/scripts/m1_1_pose_image.py`
  - Model: `yolo11n-pose.pt` (Ultralytics 8.4.171, PyTorch 2.14.1+cpu, CUDA không có)
  - Ảnh: `bus.jpg` (ảnh mẫu Ultralytics), 810×1080 pixel
  - Ngưỡng giữ một người: `--conf 0.5`
  - Ngưỡng gắn ghi chú khớp yếu trong script: 0.5
- Commit: <điền hash hoặc "chưa commit">
- Kết quả (đơn vị: 1 ảnh, 3 người; tọa độ pixel; gốc trên-trái; trục y hướng xuống):
  - Shape keypoints: (3, 17, 2)
  - Người #0 — người áo be bên trái, khoanh tay. Box confidence 0.89. Khung (48, 400) → (244, 906). Tỷ lệ rộng/cao 0.39. Mặt đối camera. Mắt cá thấp hơn hông khoảng 200 pixel. Cổ tay có y nhỏ hơn khuỷu. `left_ear` conf 0.43 (tọa độ vẫn nằm trên đầu); các khớp còn lại conf ≥ 0.90.
  - Người #1 — người áo tối bị cắt ở mép phải. Box confidence 0.88. Khung (669, 388) → (810, 878). Tỷ lệ rộng/cao 0.29. `left_shoulder` x = 810 (sát biên ảnh). Mặt conf ≤ 0.24. Hai vai cách khoảng 3.5 pixel. Chân còn trong khung: y tăng từ vai (487) tới mắt cá (844) ở bên trái người đó.
  - Người #2 — người đứng giữa. Box confidence 0.88. Khung (223, 405) → (344, 859). Tỷ lệ rộng/cao 0.27. `left_shoulder` x = 318, `right_shoulder` x = 251 (mặt đối camera, hai vai cách khoảng 67 pixel). Hầu hết khớp conf ≥ 0.91; `left_ear` conf 0.55.
- Nhận xét:
  - Confidence của khung bao chỉ nói «đây là một người». Người #1 có box 0.88 nhưng mặt và nhiều khớp dưới 0.5.
  - Khớp confidence thấp vẫn có thể nằm gần chỗ hợp lý (`left_ear` người #0), không nhất thiết về (0, 0).
  - Tỷ lệ rộng/cao < 1 khớp với ba người đang đứng trên ảnh này. Người #1 bị cắt khung nên tỷ lệ không phản ánh hết bề ngang cơ thể.
  - `bus.jpg` chỉ để học API, không phải sinh hoạt trong nhà.
- Câu hỏi mở: Confidence tối thiểu bao nhiêu thì bỏ một keypoint khi tính đặc trưng?
- Tiếp theo: Milestone 1.2 — chạy pose trên video và lưu video có khung xương.

## 2026-10-04 — Milestone 1.2