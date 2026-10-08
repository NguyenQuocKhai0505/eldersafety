"""
Milestone 1.3 — Xuất keypoint của mọi frame trong video ra file CSV (+ file metadata JSON).

Mục tiêu học tập:
- Tách việc "chạy model" (chậm, làm MỘT lần) khỏi việc "phân tích dữ liệu" (nhanh, làm NHIỀU lần).
- Thiết kế dữ liệu thô: lưu đầy đủ, KHÔNG lọc bớt ở bước xuất.
- Lưu metadata để kết quả có thể tái lập.

Cách chạy (từ thư mục research, đã kích hoạt .venv):
    python scripts/m1_3_export_keypoints.py --source ../data/m1_2/dung_yen.mp4
    python scripts/m1_3_export_keypoints.py --source ../data/m1_2/ngoi_dung.mp4 --stride 3
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import ultralytics
from ultralytics import YOLO

# Thứ tự 17 keypoint theo chuẩn COCO.
KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1.3: xuất keypoint ra CSV")
    parser.add_argument("--source", required=True, help="Đường dẫn file video")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="File model pose")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Ngưỡng confidence để giữ lại một người")
    parser.add_argument("--stride", type=int, default=1,
                        help="Chỉ xử lý 1 frame trong mỗi N frame (1 = xử lý mọi frame)")
    # Lưu trong data/ (không commit): keypoint vẫn là dữ liệu về người tham gia
    parser.add_argument("--out-dir", default="../data/keypoints",
                        help="Thư mục lưu CSV và metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, khoảng {total_reported} frame")

    rows: list[dict] = []          # mỗi phần tử = một dòng CSV
    frame_idx = -1
    frames_processed = 0
    people_count = {"0": 0, "1": 0, "2+": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride != 0:   # bỏ qua frame theo stride
            continue
        frames_processed += 1

        result = model(frame, conf=args.conf, verbose=False)[0]
        n = len(result.boxes)
        time_s = round(frame_idx / fps, 4)

        if n == 0:


            people_count["0"] += 1
            rows.append({"frame": frame_idx, "time_s": time_s, "person_idx": -1})
            continue
        people_count["1" if n == 1 else "2+"] += 1

        boxes = result.boxes.xyxy.cpu().numpy()       # (n, 4)
        box_conf = result.boxes.conf.cpu().numpy()    # (n,)
        kp_xy = result.keypoints.xy.cpu().numpy()     # (n, 17, 2)
        kp_conf = (
            result.keypoints.conf.cpu().numpy()       # (n, 17)
            if result.keypoints.conf is not None
            else np.full((n, 17), np.nan)
        )

        for i in range(n):
            # LƯU Ý: chưa có tracking, person_idx chỉ là thứ tự phát hiện trong frame,
            # KHÔNG phải danh tính cố định của một người qua các frame.
            row = {
                "frame": frame_idx,
                "time_s": time_s,
                "person_idx": i,
                "box_x1": round(float(boxes[i, 0]), 2),
                "box_y1": round(float(boxes[i, 1]), 2),
                "box_x2": round(float(boxes[i, 2]), 2),
                "box_y2": round(float(boxes[i, 3]), 2),
                "box_conf": round(float(box_conf[i]), 3),
            }
            # Lưu ĐẦY ĐỦ mọi keypoint, kể cả confidence thấp. Việc lọc để dành cho bước phân tích.
            for k, name in enumerate(KEYPOINT_NAMES):
                row[f"{name}_x"] = round(float(kp_xy[i, k, 0]), 2)
                row[f"{name}_y"] = round(float(kp_xy[i, k, 1]), 2)
                row[f"{name}_conf"] = round(float(kp_conf[i, k]), 3)
            rows.append(row)

    cap.release()

    if not rows:
        print("Không có dữ liệu nào được ghi. Kiểm tra lại video.")
        return

    # ---- Ghi CSV ----
    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem          # "dung_yen.mp4" -> "dung_yen"
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)

    # ---- Ghi metadata: mọi thông tin cần để tái lập file CSV này ----
    meta = {  
        "source": str(args.source),
        "fps": fps,
        "width": width,
        "height": height,
        "frames_reported": total_reported,
        "frames_read": frame_idx + 1,
        "frames_processed": frames_processed,
        "stride": args.stride,
        "model": args.model,
        "person_conf_threshold": args.conf,
        "ultralytics_version": ultralytics.__version__,
        "keypoint_names": KEYPOINT_NAMES,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_dir / f"{stem}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Tổng kết ----
    print("\n=== KẾT QUẢ ===")
    print(f"Frame đã xử lý      : {frames_processed} (stride = {args.stride})")
    print(f"Frame có 0 / 1 / 2+ người: {people_count['0']} / {people_count['1']} / {people_count['2+']}")
    print(f"Số dòng CSV          : {len(df)}")
    print(f"Số cột CSV           : {len(df.columns)}")
    print(f"Đã lưu: {csv_path}")
    print(f"Đã lưu: {meta_path}")


if __name__ == "__main__":
    main()