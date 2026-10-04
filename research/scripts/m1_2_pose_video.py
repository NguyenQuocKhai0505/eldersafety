"""
Milestone 1.2 — Pose estimation trên video + đo tốc độ xử lý và độ rung (jitter) của keypoint.

Mục tiêu học tập:
- Video là chuỗi frame (ảnh) liên tiếp; model xử lý TỪNG frame độc lập.
- Đo thời gian inference để biết máy có chạy kịp thời gian thực không.
- Thấy tận mắt: keypoint RUNG giữa các frame dù người đứng yên.

Cách chạy (từ thư mục research, đã kích hoạt .venv):
    python scripts/m1_2_pose_video.py --source ../data/m1_2/dung_yen.mp4
    python scripts/m1_2_pose_video.py --source 0 --show --max-frames 300     # webcam
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

LEFT_HIP = 11        # chỉ số COCO của hông trái
KP_CONF_MIN = 0.5    # chỉ dùng keypoint có confidence >= ngưỡng này


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1.2: pose estimation trên video")
    parser.add_argument("--source", required=True,
                        help="Đường dẫn video, hoặc 0 để dùng webcam")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="File model pose")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Ngưỡng confidence để giữ lại một người")
    parser.add_argument("--out", default="runs/m1_2/annotated.mp4",
                        help="Nơi lưu video kết quả")
    parser.add_argument("--show", action="store_true",
                        help="Hiện cửa sổ xem trực tiếp (nhấn q để dừng)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Dừng sau N frame (0 = chạy hết video)")
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    """Mở webcam (nếu source là số) hoặc file video."""
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {source}")
    return cap


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    cap = open_source(args.source)

    # Thông tin video. Webcam đôi khi trả về FPS = 0, khi đó tạm dùng 30.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Nguồn: {width}x{height} @ {fps:.1f} FPS")

    # Chuẩn bị ghi video kết quả. "mp4v" là codec (cách nén video) cho file .mp4.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    infer_ms: list[float] = []            # thời gian inference từng frame (ms)
    hip_xy: list[np.ndarray] = []         # tọa độ hông trái của người #0 khi thấy rõ
    frames = 0
    frames_no_person = 0

    while True:
        ok, frame = cap.read()            # frame là ảnh BGR dạng mảng NumPy (H, W, 3)
        if not ok:                        # hết video hoặc mất kết nối camera
            break
        frames += 1

        # Đo thời gian inference
        t0 = time.perf_counter()
        result = model(frame, conf=args.conf, verbose=False)[0]
        infer_ms.append((time.perf_counter() - t0) * 1000)

        # Ghi lại tọa độ hông trái của người #0 (nếu có và đủ tin cậy).
        # LƯU Ý: chưa có tracking, nên "người #0" ở các frame khác nhau
        # CHƯA CHẮC là cùng một người. Vì vậy hãy dùng video chỉ có MỘT người.
        if len(result.boxes) == 0:
            frames_no_person += 1
        elif result.keypoints.conf is not None:
            kp_xy = result.keypoints.xy.cpu().numpy()
            kp_conf = result.keypoints.conf.cpu().numpy()
            if kp_conf[0, LEFT_HIP] >= KP_CONF_MIN:
                hip_xy.append(kp_xy[0, LEFT_HIP])

        annotated = result.plot()
        writer.write(annotated)

        if args.show:
            cv2.imshow("M1.2 - press q to stop", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and frames >= args.max_frames:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if frames == 0:
        print("Không đọc được frame nào. Kiểm tra lại đường dẫn hoặc codec của video.")
        return

    # ---- Tổng kết tốc độ ----
    # Bỏ frame đầu: lần chạy đầu tiên luôn chậm hơn vì model còn đang "khởi động" (warm-up).
    timing = infer_ms[1:] if len(infer_ms) > 1 else infer_ms
    mean_ms = float(np.mean(timing))
    median_ms = float(np.median(timing))
    print("\n=== TỐC ĐỘ ===")
    print(f"Số frame đã xử lý      : {frames}")
    print(f"Inference trung bình   : {mean_ms:.1f} ms/frame (trung vị {median_ms:.1f} ms)")
    print(f"FPS xử lý ước tính     : {1000 / mean_ms:.1f}")
    print(f"Khoảng cách giữa 2 frame của video: {1000 / fps:.1f} ms")

    # ---- Tổng kết độ rung ----
    print("\n=== PHÁT HIỆN & ĐỘ RUNG (hông trái, người #0) ===")
    print(f"Frame không thấy người : {frames_no_person} / {frames}")
    print(f"Frame thấy rõ hông trái: {len(hip_xy)} / {frames}")
    if len(hip_xy) >= 2:
        hip = np.array(hip_xy)                     # shape (số frame, 2)
        std_x, std_y = hip.std(axis=0)             # độ lệch chuẩn theo x và y
        range_x, range_y = hip.max(axis=0) - hip.min(axis=0)
        print(f"Độ lệch chuẩn (pixel)  : x = {std_x:.2f}, y = {std_y:.2f}")
        print(f"Biên độ max-min (pixel): x = {range_x:.1f}, y = {range_y:.1f}")
        y = hip[:, 1]
        buoc_y = np.abs(np.diff(y))
        trung_vi_y = float(np.median(buoc_y))
        print(f"Trung vị |y[t]-y[t-1]| : {trung_vi_y:.2f} pixel/frame")
        print("(Chỉ có ý nghĩa là 'độ rung' nếu người ĐỨNG YÊN trong suốt video.)")

    print(f"\nĐã lưu video kết quả: {out_path}")


if __name__ == "__main__":
    main()