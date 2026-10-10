"""
Milestone 2.2 — Làm sạch chuỗi keypoint của MỘT người: loại điểm không tin cậy,
lấp khoảng trống ngắn, làm mượt.
 
Đầu vào : CSV có track_id của M2.1 (vd ngoi_dung_track.csv) + file .meta.json đi kèm.
Đầu ra  : <stem>_smooth.csv, <stem>_smooth.meta.json và một biểu đồ thô vs mượt.
 
Script này KHÔNG chạy model, chỉ đọc CSV nên chạy rất nhanh và chạy lại bao nhiêu lần cũng được.
 
Cách chạy (từ thư mục research, đã kích hoạt .venv):
    python scripts/m2_2_smooth.py --csv ../data/keypoints/dung_yen_track.csv
    python scripts/m2_2_smooth.py --csv ../data/keypoints/ngoi_dung_track.csv --method mean
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
KEYPOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
METHODS: list[str] = ["median", "mean", "ema"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2.2: làm sạch và làm mượt keypoint của một track")
    parser.add_argument("--csv", required=True, help="CSV có track_id từ M2.1")
    parser.add_argument("--track-id", type=int, default=None,
                        help="ID cần xử lý. Bỏ trống = lấy ID xuất hiện nhiều frame nhất")
    parser.add_argument("--kp-conf", type=float, default=0.5,
                        help="Keypoint có confidence thấp hơn ngưỡng này bị coi là thiếu")
    # Tham số tính bằng GIÂY chứ không phải số frame, để vẫn đúng khi edge chạy 5–10 FPS.
    parser.add_argument("--max-gap-s", type=float, default=0.2,
                        help="Chỉ lấp khoảng trống ngắn hơn hoặc bằng chừng này giây")
    parser.add_argument("--method", choices=METHODS, default="median", help="Cách làm mượt")
    parser.add_argument("--window-s", type=float, default=0.2, help="Độ dài cửa sổ làm mượt (giây)")
    parser.add_argument("--out-dir", default="../data/keypoints", help="Thư mục lưu CSV và metadata")
    parser.add_argument("--plot-dir", default="runs/m2_2", help="Thư mục lưu biểu đồ")
    parser.add_argument("--overwrite", action="store_true", help="Cho phép ghi đè file đầu ra đã có")
    return parser.parse_args()
 
def select_track(df: pd.DataFrame, track_id: int | None) -> tuple[pd.DataFrame, int]:
    """Lấy các dòng của một track, mỗi frame đã xử lý đúng một dòng (frame vắng mặt -> NaN)."""
    people = df[df["track_id"] >= 0]
    if people.empty:
        raise SystemExit("CSV không có track nào (mọi track_id đều là -1).")

    if track_id is None:
        track_id = int(people["track_id"].value_counts().idxmax())   # ID có nhiều dòng nhất

    track = people[people["track_id"] == track_id].set_index("frame").sort_index()

    if track.empty:
        raise SystemExit(f"Không có track_id = {track_id} trong CSV.")
    
    if track.index.has_duplicates:
        raise SystemExit(f"track_id = {track_id} xuất hiện hai lần trong cùng một frame.")

    # M2.1 luôn ghi ít nhất một dòng cho mỗi frame đã xử lý, nên đây là danh sách frame đầy đủ.
    all_frames = np.sort(df["frame"].unique())
    # reindex: frame nào track vắng mặt thì thành một dòng toàn NaN, thay vì biến mất.
    return track.reindex(all_frames), track_id
 


def mask_unreliable(track: pd.DataFrame, kp_conf: float) -> pd.DataFrame:
    """Đổi toạ độ của keypoint không đủ tin cậy thành NaN. Cột confidence giữ nguyên."""
    track = track.copy()
    for name in KEYPOINT_NAMES:
        # Viết ~(conf >= ngưỡng) thay vì (conf < ngưỡng) để NaN (frame vắng mặt) cũng bị tính là thiếu:
        # mọi phép so sánh với NaN đều ra False.
        unreliable = ~(track[f"{name}_conf"] >= kp_conf)
        track.loc[unreliable, [f"{name}_x", f"{name}_y"]] = np.nan
    return track

def fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Nội suy tuyến tính các khoảng NaN dài <= max_gap frame. Khoảng dài hơn giữ nguyên NaN."""
    is_nan = series.isna()
    # Đánh số các khoảng trống: cumsum (tổng cộng dồn) chỉ tăng ở điểm có giá trị,
    # nên các NaN liền nhau nhận cùng một số -> cùng một nhóm.
    gap_id = (~is_nan).cumsum()
    gap_len = is_nan.groupby(gap_id).transform("sum")     # độ dài khoảng trống chứa mỗi điểm
    # limit_area="inside": chỉ nội suy giữa hai điểm có thật, không đoán ở đầu và cuối chuỗi.
    filled = series.interpolate(limit_area="inside")
    filled[is_nan & (gap_len > max_gap)] = np.nan         # trả khoảng trống dài về NaN
    return filled

def smooth(series: pd.Series, method: str, window: int) -> pd.Series:
    """Làm mượt chỉ bằng frame hiện tại và các frame TRƯỚC nó (edge chạy trực tiếp cũng làm được)."""
    if method == "median":
        smoothed = series.rolling(window, min_periods=1).median()   # trung vị của `window` frame gần nhất
    elif method == "mean":
        smoothed = series.rolling(window, min_periods=1).mean()     # trung bình của `window` frame gần nhất
    else:
        # EMA (trung bình trượt hàm mũ): mới = alpha * hiện_tại + (1 - alpha) * cũ, alpha = 2 / (window + 1)
        smoothed = series.ewm(span=window, adjust=False).mean()
    return smoothed.where(series.notna())    # chỗ nào đang thiếu thì vẫn để thiếu



def clean_keypoints(track: pd.DataFrame, max_gap: int, method: str, window: int) -> pd.DataFrame:
    """Lấp khoảng trống ngắn rồi làm mượt cho mọi cột toạ độ keypoint."""
    track = track.copy()
    for name in KEYPOINT_NAMES:
        for axis in ("x", "y"):
            column = f"{name}_{axis}"
            track[column] = smooth(fill_short_gaps(track[column], max_gap), method, window)
    return track
 
 
def hip_center_y(track: pd.DataFrame) -> pd.Series:
    return (track["left_hip_y"] + track["right_hip_y"]) / 2
 
 
def jitter_stats(series: pd.Series) -> dict:
    """|Δy| giữa hai frame liên tiếp (cùng cách đo với M1.4): trung vị, P95, max."""
    steps = series.diff().abs().dropna()
    if steps.empty:
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "median": round(float(steps.median()), 2),
        "p95": round(float(steps.quantile(0.95)), 2),
        "max": round(float(steps.max()), 2),
    }
 
 
def count_points(track: pd.DataFrame) -> int:
    """Số điểm keypoint đang có toạ độ (tối đa 17 x số frame)."""
    return int(sum(track[f"{name}_y"].notna().sum() for name in KEYPOINT_NAMES))
 
 
def plot_hip(time_s: pd.Series, raw: pd.Series, smoothed: pd.Series, title: str, path: Path) -> None:
    """Vẽ y tâm hông thô và đã làm mượt trên cùng một trục."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time_s, raw, color="#2a78d6", linewidth=1.0, label="Thô (đã lọc confidence)")
    ax.plot(time_s, smoothed, color="#eb6834", linewidth=2.0, label="Sau M2.2")
    ax.invert_yaxis()                      # y ảnh hướng xuống -> đảo để "xuống" là hạ người
    ax.set_xlabel("Thời gian (giây)")
    ax.set_ylabel("y tâm hông (pixel)")
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def main() -> None:
    # Cửa sổ lệnh Windows mặc định không in được tiếng Việt.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
 
    csv_path = Path(args.csv)
    source_meta_path = csv_path.with_suffix(".meta.json")     # ngoi_dung_track.meta.json
    if not csv_path.exists() or not source_meta_path.exists():
        raise SystemExit(f"Cần cả {csv_path} và {source_meta_path}.")
 
    stem = csv_path.stem.removesuffix("_track")               # "ngoi_dung_track" -> "ngoi_dung"
    out_csv = Path(args.out_dir) / f"{stem}_smooth.csv"
    out_meta = Path(args.out_dir) / f"{stem}_smooth.meta.json"
    out_plot = Path(args.plot_dir) / f"{stem}_hip_smooth.png"
    existing = [str(p) for p in (out_csv, out_meta, out_plot) if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Đã tồn tại: {existing}. Thêm --overwrite nếu chắc chắn muốn ghi đè.")
 
    # FPS lấy từ metadata của M2.1, không gõ tay: đây là lý do M1.3 lưu file meta.
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    fps = source_meta["fps"] / source_meta["stride"]          # FPS thực của chuỗi đã xử lý
    max_gap = max(1, round(args.max_gap_s * fps))             # đổi giây -> số frame
    window = max(1, round(args.window_s * fps))
 
    df = pd.read_csv(csv_path)
    track, track_id = select_track(df, args.track_id)
    frames_absent = int(track["track_id"].isna().sum())
    points_before = count_points(track)
 
    masked = mask_unreliable(track, args.kp_conf)
    points_reliable = count_points(masked)
    # Thử từng cách làm mượt để so sánh; bản được lưu là bản của --method.
    cleaned_by_method = {m: clean_keypoints(masked, max_gap, m, window) for m in METHODS}
    cleaned = cleaned_by_method[args.method]
    points_after = count_points(cleaned)
 
    # ---- Ghi CSV: một dòng mỗi frame; cột box_* và *_conf giữ nguyên giá trị gốc ----
    cleaned = cleaned.drop(columns=["person_idx"]).reset_index()   # frame quay lại thành cột
    cleaned["time_s"] = (cleaned["frame"] / source_meta["fps"]).round(4)
    cleaned["track_id"] = track_id
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_csv, index=False, float_format="%.2f")
 
    # ---- Đo độ rung của tâm hông trước và sau ----
    raw_hip = hip_center_y(masked)
    stats = {"raw": jitter_stats(raw_hip)}
    for m in METHODS:
        stats[m] = jitter_stats(hip_center_y(cleaned_by_method[m]))
 
    plot_hip(cleaned["time_s"], raw_hip.to_numpy(), hip_center_y(cleaned),
             f"M2.2 — {stem}: y tâm hông, {args.method}, cửa sổ {window} frame", out_plot)
 
    meta = {
        "source_csv": str(csv_path),
        "track_id": track_id,
        "fps": fps,
        "kp_conf_threshold": args.kp_conf,
        "max_gap_s": args.max_gap_s,
        "max_gap_frames": max_gap,
        "method": args.method,
        "window_s": args.window_s,
        "window_frames": window,
        "frames": len(cleaned),
        "frames_track_absent": frames_absent,
        "points_total": 17 * len(cleaned),
        "points_in_source": points_before,
        "points_reliable": points_reliable,
        "points_filled": points_after - points_reliable,
        "points_still_missing": 17 * len(cleaned) - points_after,
        "hip_center_abs_dy_px": stats,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
 
    # ---- Tổng kết ----
    print("\n=== KẾT QUẢ ===")
    print(f"Track ID {track_id}: {len(cleaned)} frame, vắng mặt {frames_absent} frame")
    print(f"FPS {fps:.1f} -> lấp khoảng trống <= {max_gap} frame, cửa sổ làm mượt {window} frame")
    print(f"Điểm keypoint (17 x {len(cleaned)} = {17 * len(cleaned)}):")
    print(f"  bị loại vì confidence < {args.kp_conf} : {points_before - points_reliable}")
    print(f"  được nội suy                  : {points_after - points_reliable}")
    print(f"  vẫn còn thiếu                 : {17 * len(cleaned) - points_after}")
    print("Độ rung |dy| của tâm hông giữa hai frame liên tiếp (pixel):")
    print(f"  {'':<8}{'trung vị':>10}{'P95':>8}{'max':>8}")
    for name, s in stats.items():
        chosen = "  <- đã lưu" if name == args.method else ""
        print(f"  {name:<8}{s['median']:>10.2f}{s['p95']:>8.2f}{s['max']:>8.2f}{chosen}")
    print(f"Đã lưu: {out_csv}")
    print(f"Đã lưu: {out_meta}")
    print(f"Đã lưu: {out_plot}")
 
 
if __name__ == "__main__":
    main()