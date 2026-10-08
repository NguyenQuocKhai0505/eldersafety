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
- Mục tiêu: Chạy YOLO11n-pose trên video, lưu video có khung xương, đo thời gian inference và độ dịch của hông trái (chỉ số 11) giữa các frame. Chưa phát hiện té ngã.
- Cấu hình:
  - Script: `research/scripts/m1_2_pose_video.py`
  - Model: `yolo11n-pose.pt` (Ultralytics 8.4.171, PyTorch 2.14.1+cpu, CUDA không có)
  - Ngưỡng giữ một người: `--conf 0.5`
  - Khớp dùng: hông trái, chỉ số COCO 11, chỉ khi confidence khớp ≥ 0.5
  - Video: `data/m1_2/dung_yen.mp4` và `data/m1_2/ngoi_dung.mp4`, cả hai 720×1280, 30 FPS (mỗi frame cách 33.3 ms)
  - Một người trong khung, quay chính diện, trong nhà
  - Codec file kết quả: `mp4v` (Cursor không phát được file này)
- Commit: chưa commit
- Kết quả (đơn vị: pixel trên ảnh 720×1280; gốc trên-trái; trục y hướng xuống; người #0 từng frame, chưa có tracking):
  - Đứng yên — 460 frame (khoảng 15.3 giây). Inference trung bình 35.3 ms/frame (trung vị 34.9 ms), khoảng 28.3 frame/giây. Frame không thấy người: 0/460. Frame thấy rõ hông trái: 460/460. Độ lệch chuẩn hông: x = 10.40, y = 4.86. Biên độ max−min: x = 64.9, y = 27.3. Trung vị `|y[t]−y[t−1]|` = 0.33 pixel/frame (N = 460, nên có 459 bước).
  - Ngồi rồi đứng — 523 frame (khoảng 17.4 giây). Inference trung bình 36.7 ms/frame (trung vị 35.4 ms), khoảng 27.3 frame/giây. Frame không thấy người: 0/523. Frame thấy rõ hông trái: 523/523. Độ lệch chuẩn hông: x = 61.87, y = 143.57. Biên độ max−min: x = 292.2, y = 485.8. Trung vị `|y[t]−y[t−1]|`: chưa đo (lần chạy này xong trước khi sửa `y = hip[:, 1]`).
  - Công thức: độ lệch chuẩn của NumPy chia cho số frame N. Biên độ = max − min của cả clip. Trung vị bước = median của `|y[t]−y[t−1]|`.
- Nhận xét:
  - Cả hai clip đều thấy người và thấy hông trái ở mọi frame. Khung xương vẽ đúng trên người đang đứng.
  - `y` đứng yên ổn định (độ lệch chuẩn 4.86 pixel, biên độ 27.3). `y` lúc ngồi rồi đứng lớn (143.57 và 485.8) vì hông đổi độ cao thật, không phải vì model mất người.
  - Trung vị 0.33 pixel/frame trên clip đứng yên là bước nhảy điển hình giữa hai frame. Biên độ 27.3 pixel là chênh cả clip. Hai số này không thay nhau.
  - Inference quanh 33–37 ms/frame, gần với 33.3 ms của video 30 FPS. Máy này xử lý khoảng 27–28 frame/giây.
  - Chưa có tracking. Số hông chỉ có nghĩa khi trong khung có một người.
- Câu hỏi mở: Trung vị `|y[t]−y[t−1]|` của clip ngồi–đứng có vẫn nhỏ không, vì phần lớn frame vẫn đứng hoặc ngồi yên?
- Tiếp theo: chạy lại `ngoi_dung.mp4` để có trung vị bước nhảy, ghi số đó vào mục này, rồi sang Milestone 1.3 (xuất keypoint từng frame ra CSV).

## 2026-10-05 — Milestone 1.3
- Mục tiêu: Xuất keypoint mọi frame của một video ra CSV, kèm JSON metadata, để lần sau phân tích không phải chạy lại model. Chưa đo té ngã.
- Cấu hình:
  - Script: `research/scripts/m1_3_export_keypoints.py`
  - Model: `yolo11n-pose.pt` (Ultralytics 8.4.171, PyTorch 2.14.1+cpu, CUDA không có)
  - Video: `data/m1_2/dung_yen.mp4`, 720×1280, 30 FPS
  - `--conf 0.5`, `--stride 1`
  - File ra: `data/keypoints/dung_yen.csv`, `data/keypoints/dung_yen.meta.json`
- Commit: chưa commit
- Kết quả (đơn vị: 1 video, 460 frame; tọa độ pixel; gốc trên-trái; trục y hướng xuống; một dòng CSV = một người trong một frame):
  - `frames_reported` = `frames_read` = `frames_processed` = 460. Không mất frame.
  - Frame có 0 / 1 / 2+ người: 0 / 460 / 0. Số dòng CSV = 460. Số cột = 59 (8 cột chung + 17 khớp × 3).
  - `time_s` từ 0 đến 15,3. Khoảng cách hai dòng liên tiếp = 1/30 ≈ 0,0333 giây.
  - Frame 0: khung bao (225,59, 289,87) → (505,55, 1179,35), box confidence 0,923, tỷ lệ rộng/cao ≈ 0,31. `left_hip_y` = 717,24, `left_hip_conf` = 1,0.
  - Frame 459: `left_hip_y` = 708,28, `time_s` = 15,3. Mắt cá còn trong ảnh (y khoảng 1119–1128, ảnh cao 1280).
  - Năm frame đầu, `left_hip_y` lệch khoảng 1–3 pixel giữa hai frame. `right_ear_conf` frame 3 = 0,64, trong khi hông vẫn 0,999.
- Nhận xét:
  - CSV và JSON khớp nhau: 460 frame, stride 1, ngưỡng 0,5, model và phiên bản Ultralytics đúng như lúc chạy.
  - Không có dòng `person_idx = -1`. Clip này model không mất người. Dòng đó vẫn cần cho clip sau, để chỗ mất người còn nằm trên trục thời gian.
  - `person_idx` luôn là 0 vì trong khung chỉ có một người. Đây chưa phải tracking.
  - Hông trái đổi khoảng 9 pixel sau 15,3 giây (717 → 708). y nhỏ hơn nghĩa là hông cao hơn trên ảnh. Mức này là rung và nhích nhẹ, không phải cú hạ người.
  - Tỷ lệ khung bao khoảng 0,31, khớp tư thế đứng. Khớp confidence thấp (tai) vẫn được ghi, không bị bỏ ở bước xuất.
- Câu hỏi mở: Với clip ngồi–đứng, `left_hip_y` tăng thêm bao nhiêu pixel so với khoảng 9 pixel của clip đứng yên?
- Tiếp theo: xuất `ngoi_dung.mp4` bằng cùng script, stride 1, rồi sang Milestone 1.4 — vẽ `left_hip_y` theo `time_s` của hai clip.


## 2026-10-06 — Milestone 1.4
- Mục tiêu: Vẽ `y` hông và tốc độ `|dy/dt|` của hai clip để thấy nhiễu lúc đứng yên và cú hạ người lúc ngồi. Chưa phát hiện té ngã.
- Cấu hình:
  - Script: `research/scripts/m1_4_plot_hip.py`
  - CSV: `data/keypoints/dung_yen.csv` (460 dòng), `data/keypoints/ngoi_dung.csv` (523 dòng)
  - Ngưỡng keypoint: `--kp-conf 0.5`
  - Người giữ lại: `person_idx = 0`
  - Tâm hông = `(left_hip_y + right_hip_y) / 2`, chỉ khi cả hai confidence ≥ 0.5
  - `|Δy|` = trị tuyệt đối của `np.diff` sau khi bỏ NaN. Tốc độ = `|Δy| / Δt`, đơn vị pixel/giây
  - Ảnh: `research/runs/m1_4/dung_yen_hip.png`, `research/runs/m1_4/ngoi_dung_hip.png`
- Commit: chưa commit
- Kết quả (đơn vị: pixel trên ảnh 720×1280; gốc trên-trái; trục y hướng xuống; một bước = hai frame liên tiếp của người #0):
  - Cả hai clip: 0 điểm hông trái và 0 điểm tâm hông bị loại vì confidence thấp.
  - Đứng yên, 460 frame, 459 bước. Hông trái `|Δy|`: trung vị 0,34 px, P95 2,35 px, max 9,20 px. Tốc độ: trung vị 10 px/s, P95 70 px/s, max 276 px/s. Tâm hông `|Δy|`: trung vị 0,31 px, P95 2,35 px, max 8,79 px. Tốc độ: trung vị 9 px/s, P95 71 px/s, max 264 px/s.
  - Ngồi–đứng, 523 frame, 522 bước. Hông trái `|Δy|`: trung vị 2,61 px, P95 20,26 px, max 37,82 px. Tốc độ: trung vị 78 px/s, P95 607 px/s, max 1132 px/s. Tâm hông `|Δy|`: trung vị 2,47 px, P95 20,42 px, max 37,67 px. Tốc độ: trung vị 74 px/s, P95 613 px/s, max 1128 px/s.
  - Tâm hông cả clip ngồi–đứng: min 649 px, max 1120 px, biên độ 471 px.
  - Cửa sổ dưới đây ước lượng bằng mắt trên biểu đồ rồi đo lại trên CSV, chưa phải nhãn sự kiện. Tốc độ là của tâm hông:
    - 0–2,5 s, hạ lần 1: y khoảng 700–844, trung vị 77 px/s, max 333 px/s.
    - 2,5–6 s, ngồi tương đối yên: y 832–853, trung vị 17 px/s, max 123 px/s.
    - 6–7,5 s, đứng dậy: trung vị 74 px/s, max 287 px/s.
    - 7,5–9,5 s, đứng: y 715–734, trung vị 40 px/s, max 227 px/s.
    - 9,5–11,5 s, hạ lần 2: y khoảng 710–1078, trung vị 281 px/s, max 1128 px/s.
    - 11,5–14 s, ở dưới: y 1066–1120, trung vị 154 px/s, max 1100 px/s.
    - 14–16 s, đứng dậy: trung vị 305 px/s, max 1038 px/s.
- Nhận xét:
  - Trung vị 0,34 px/bước lúc đứng yên khớp mức 0,33 px/frame của Milestone 1.2. CSV làm tròn tọa độ 2 chữ số thập phân, nên hai số lệch nhẹ.
  - Câu hỏi mở của 1.2 đã có số: trung vị `|Δy|` của cả clip ngồi–đứng là 2,61 px, không còn nhỏ. Hơn nửa số frame nằm trong lúc hạ hoặc đứng dậy.
  - Lần ngồi thứ hai đạt 1128 px/s, cao hơn max 276 px/s lúc đứng yên. Tốc độ một mình không tách được ngồi xuống khỏi một ngưỡng lấy từ clip đứng yên.
  - Đoạn 2,5–6 s tốc độ về khoảng 17 px/s, gần mặt bằng nhiễu. Đoạn 11,5–14 s hông thấp nhưng trung vị vẫn 154 px/s, nên chưa gọi là bất động. Chưa xem video để phân biệt người còn cựa với khung xương rung.
  - Tương quan bước nhảy `y` giữa hai hông trên clip đứng yên là 0,91. Tâm hông vì vậy chỉ ổn hơn từng hông một chút trên clip này.
  - Hai clip chưa phải dữ liệu té ngã.