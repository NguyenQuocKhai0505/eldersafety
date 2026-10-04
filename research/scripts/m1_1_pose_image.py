"""
Milestone 1.1 — Chạy pose estimation trên một ảnh tĩnh.

Mục tiêu học tập:
- Hiểu model trả về gì: khung bao (bounding box), 17 keypoint, confidence.
- Hiểu hệ tọa độ ảnh: gốc (0, 0) ở góc TRÊN-TRÁI, trục y hướng XUỐNG.

Cách chạy (từ thư mục research, đã kích hoạt .venv):
    python scripts/m1_1_pose_image.py --image https://ultralytics.com/images/bus.jpg
    python scripts/m1_1_pose_image.py --image "duong/dan/anh_cua_ban.jpg"
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# Thứ tự 17 keypoint theo chuẩn COCO.
# "left"/"right" là trái/phải CỦA NGƯỜI trong ảnh, không phải của người xem.
KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1.1: pose estimation trên ảnh tĩnh")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh hoặc URL")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="File model pose")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Ngưỡng confidence để giữ lại một người")
    # Lưu vào runs/ (đã nằm trong .gitignore) để ảnh có người không bị commit lên Git
    parser.add_argument("--out", default="runs/m1_1/annotated.jpg",
                        help="Nơi lưu ảnh kết quả")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Nạp model (lần đầu sẽ tự tải file .pt)
    model = YOLO(args.model)

    # 2. Chạy inference. Kết quả là một list, mỗi phần tử ứng với một ảnh.
    results = model(args.image, conf=args.conf, verbose=False)
    result = results[0]  # ta chỉ đưa vào 1 ảnh

    height, width = result.orig_shape
    print(f"Kích thước ảnh: {width} x {height} (rộng x cao)")

    # 3. Lấy dữ liệu ra dạng NumPy để dễ đọc.
    #    .cpu()   : chuyển dữ liệu về CPU (phòng khi chạy trên GPU)
    #    .numpy() : đổi tensor của PyTorch thành mảng NumPy
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # shape (N, 4): x1, y1, x2, y2
    box_conf = result.boxes.conf.cpu().numpy()     # shape (N,)
    kp_xy = result.keypoints.xy.cpu().numpy()      # shape (N, 17, 2)
    kp_conf = (
        result.keypoints.conf.cpu().numpy()        # shape (N, 17)
        if result.keypoints.conf is not None
        else None
    )

    num_people = len(boxes_xyxy)
    print(f"Số người phát hiện được: {num_people}")
    print(f"Shape keypoints: {kp_xy.shape}  -> (số người, 17 khớp, [x, y])\n")

    # 4. In thông tin từng người
    for i in range(num_people):
        x1, y1, x2, y2 = boxes_xyxy[i]
        box_w, box_h = x2 - x1, y2 - y1
        print(f"=== Người #{i} | confidence = {box_conf[i]:.2f} ===")
        print(f"Khung bao: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f}) | "
              f"tỷ lệ rộng/cao = {box_w / box_h:.2f}")

        for k, name in enumerate(KEYPOINT_NAMES):
            x, y = kp_xy[i, k]
            c = kp_conf[i, k] if kp_conf is not None else float("nan")
            # Khớp không nhìn thấy thường có tọa độ gần (0, 0) và confidence thấp
            note = "  <- có thể bị che / không thấy" if c < 0.5 else ""
            print(f"  {k:2d} {name:<15} x={x:7.1f}  y={y:7.1f}  conf={c:.2f}{note}")
        print()

    # 5. Lưu ảnh đã vẽ khung xương. result.plot() trả về ảnh dạng BGR (chuẩn của OpenCV).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result.plot())
    print(f"Đã lưu ảnh kết quả: {out_path}")


if __name__ == "__main__":
    main()