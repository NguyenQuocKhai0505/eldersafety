"""
Milestone 2.1 — Tracking: gắn track ID cho từng người qua các frame, xuất CSV (+ metadata JSON).

Khác M1.3: gọi model.track, CSV thêm cột track_id, file ra tên <stem>_track.csv
để không đè CSV cũ. Script từ chối ghi đè file đã có, trừ khi có --overwrite.

Cách chạy (từ thư mục research, đã kích hoạt .venv):
    python scripts/m2_1_track.py --source ../data/m1_2/ngoi_dung.mp4
    python scripts/m2_1_track.py --source ../data/m1_2/ngoi_dung.mp4 --save-video --overwrite
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import ultralytics
from ultralytics import YOLO

# Thứ tự 17 keypoint theo chuẩn COCO (giống M1.3).
KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# track_id = -1 có hai trường hợp, phân biệt bằng person_idx:
#   person_idx = -1 : frame không có người (giống quy ước của M1.3)
#   person_idx >= 0 : model thấy người nhưng tracker chưa xác nhận track
NO_TRACK_ID = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2.1: pose + tracking, xuất CSV có track_id")
    parser.add_argument("--source", required=True, help="Đường dẫn file video")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="File model pose")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="File cấu hình tracker: bytetrack.yaml hoặc botsort.yaml")
    # Thấp hơn 0.5 của M1.3: ByteTrack dùng cả phát hiện yếu để giữ track khi người bị che.
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Ngưỡng confidence của người đưa vào tracker")
    parser.add_argument("--stride", type=int, default=1,
                        help="Chỉ xử lý 1 frame trong mỗi N frame. Tracking nên để 1.")
    parser.add_argument("--out-dir", default="../data/keypoints",
                        help="Thư mục lưu CSV và metadata")
    parser.add_argument("--save-video", action="store_true",
                        help="Lưu thêm video có vẽ khung xương và ID để kiểm tra bằng mắt")
    # Video có hình người nên lưu trong data/ (không commit), không lưu trong research/runs/.
    parser.add_argument("--video-dir", default="../data/m2_1",
                        help="Thư mục lưu video kiểm tra (chỉ dùng khi có --save-video)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Cho phép ghi đè file đầu ra đã có")
    return parser.parse_args()


def stop_if_exists(paths: list[Path], overwrite: bool) -> None:
    """Dừng sớm nếu file đầu ra đã có, để không mất dữ liệu cũ."""
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise SystemExit(f"Đã tồn tại: {existing}. Thêm --overwrite nếu chắc chắn muốn ghi đè.")


def result_to_rows(result, frame_idx: int, time_s: float) -> list[dict]:
    """Đổi kết quả của một frame thành các dòng CSV (mỗi người một dòng)."""
    n = len(result.boxes)
    if n == 0:
        return [{"frame": frame_idx, "time_s": time_s, "person_idx": -1, "track_id": NO_TRACK_ID}]

    boxes = result.boxes.xyxy.cpu().numpy()       # (n, 4)
    box_conf = result.boxes.conf.cpu().numpy()    # (n,)
    kp_xy = result.keypoints.xy.cpu().numpy()     # (n, 17, 2)
    kp_conf = (
        result.keypoints.conf.cpu().numpy()       # (n, 17)
        if result.keypoints.conf is not None
        else np.full((n, 17), np.nan)
    )
    # boxes.id là None khi tracker chưa xác nhận track nào trong frame này.
    track_ids = (
        result.boxes.id.cpu().numpy()             # (n,)
        if result.boxes.id is not None
        else np.full(n, NO_TRACK_ID)
    )

    rows: list[dict] = []
    for i in range(n):
        row = {
            "frame": frame_idx,
            "time_s": time_s,
            "person_idx": i,                  # vẫn giữ để so sánh với track_id
            "track_id": int(track_ids[i]),    # cột mới của M2.1
            "box_x1": round(float(boxes[i, 0]), 2),
            "box_y1": round(float(boxes[i, 1]), 2),
            "box_x2": round(float(boxes[i, 2]), 2),
            "box_y2": round(float(boxes[i, 3]), 2),
            "box_conf": round(float(box_conf[i]), 3),
        }
        for k, name in enumerate(KEYPOINT_NAMES):
            row[f"{name}_x"] = round(float(kp_xy[i, k, 0]), 2)
            row[f"{name}_y"] = round(float(kp_xy[i, k, 1]), 2)
            row[f"{name}_conf"] = round(float(kp_conf[i, k]), 3)
        rows.append(row)
    return rows


def summarize_tracks(df: pd.DataFrame) -> list[dict]:
    """Mỗi track ID: có bao nhiêu dòng, từ frame nào tới frame nào."""
    people = df[df["person_idx"] >= 0]  # bỏ dòng frame không có người
    if people.empty:
        return []
    # groupby ≈ GROUP BY trong SQL: gom theo track_id rồi đếm / lấy min / max của cột frame.
    grouped = people.groupby("track_id")["frame"].agg(["count", "min", "max"])
    return [
        {
            "track_id": int(track_id),
            "rows": int(stats["count"]),
            "first_frame": int(stats["min"]),
            "last_frame": int(stats["max"]),
        }
        for track_id, stats in grouped.iterrows()
    ]


def main() -> None:
    args = parse_args()

    stem = Path(args.source).stem                  # "dung_yen.mp4" -> "dung_yen"
    out_dir = Path(args.out_dir)
    csv_path = out_dir / f"{stem}_track.csv"
    meta_path = out_dir / f"{stem}_track.meta.json"
    video_path = Path(args.video_dir) / f"{stem}_track.mp4"
    outputs = [csv_path, meta_path] + ([video_path] if args.save_video else [])
    stop_if_exists(outputs, args.overwrite)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, khoảng {total_reported} frame")

    model = YOLO(args.model)

    writer = None
    if args.save_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps / args.stride, (width, height))

    rows: list[dict] = []
    frame_idx = -1
    frames_processed = 0
    track_seconds = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride != 0:
            continue
        frames_processed += 1

        # persist=True: giữ trạng thái tracker giữa các lần gọi.
        # Thiếu nó, mỗi frame tracker bị tạo lại từ đầu và ID không nối được với frame trước.
        started = time.perf_counter()
        result = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            verbose=False,
        )[0]
        track_seconds += time.perf_counter() - started

        rows.extend(result_to_rows(result, frame_idx, round(frame_idx / fps, 4)))
        if writer is not None:
            writer.write(result.plot())   # plot() trả về ảnh đã vẽ khung xương + ID

    cap.release()
    if writer is not None:
        writer.release()

    if not rows:
        print("Không có dữ liệu nào được ghi. Kiểm tra lại video.")
        return

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    tracks = summarize_tracks(df)
    ms_per_frame = track_seconds / frames_processed * 1000
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
        "tracker": args.tracker,
        "person_conf_threshold": args.conf,
        "ms_per_frame": round(ms_per_frame, 1),
        "tracks": tracks,
        "ultralytics_version": ultralytics.__version__,
        "keypoint_names": KEYPOINT_NAMES,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    frames_without_person = int((df["person_idx"] == -1).sum())
    print("\n=== KẾT QUẢ ===")
    print(f"Frame đã xử lý        : {frames_processed} (stride = {args.stride})")
    print(f"Frame không có người  : {frames_without_person}")
    print(f"Thời gian pose + track: {ms_per_frame:.1f} ms/frame (trung bình, frame đầu thường chậm)")
    print(f"Số dòng / số cột CSV  : {len(df)} / {len(df.columns)}")
    for track in tracks:
        label = "Chưa có ID" if track["track_id"] == NO_TRACK_ID else f"ID {track['track_id']}"
        print(f"  {label}: {track['rows']} dòng, frame {track['first_frame']} -> {track['last_frame']}")
    print(f"Đã lưu: {csv_path}")
    print(f"Đã lưu: {meta_path}")
    if writer is not None:
        print(f"Đã lưu: {video_path} (có hình người, KHÔNG commit)")


if __name__ == "__main__":
    main()
