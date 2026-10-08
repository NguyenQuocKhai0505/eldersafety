"""
Milestone 1.4 — Vẽ tọa độ hông và tốc độ hông theo thời gian từ file CSV keypoint.

Mục tiêu học tập:
- Đọc và lọc dữ liệu bằng pandas.
- Nhìn thấy sự kiện "ngồi xuống / đứng dậy" dưới dạng đường đồ thị.
- Đo nhiễu giữa hai frame liên tiếp (np.diff) và tốc độ (pixel/giây).
- Chẩn đoán độ rung: nhiễu nhanh, trôi chậm, hay nhảy đột ngột?

Cách chạy (từ thư mục research, đã kích hoạt .venv):
    python scripts/m1_4_plot_hip.py --csv ../data/keypoints/dung_yen.csv ../data/keypoints/ngoi_dung.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Chỉ lưu ảnh ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1.4: Vẽ tọa độ và tốc độ hông theo thời gian")
    parser.add_argument("--csv", nargs="+", required=True, help="Đường dẫn đến file CSV keypoint")
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.5,
        help="Chỉ dùng keypoint có confidence >= ngưỡng này",
    )
    parser.add_argument("--out-dir", default="runs/m1_4", help="Thư mục lưu kết quả")
    return parser.parse_args()


def load_hip_series(csv_path: Path, kp_conf_min: float) -> pd.DataFrame:
    """Đọc CSV, giữ người #0, tính tọa độ hông trái và tâm hông (khớp kém tin cậy -> NaN)."""
    df = pd.read_csv(csv_path)
    df = df[df["person_idx"] == 0].copy()

    ok_left = df["left_hip_conf"] >= kp_conf_min
    ok_right = df["right_hip_conf"] >= kp_conf_min

    # np.where(điều_kiện, giá_trị_nếu_đúng, giá_trị_nếu_sai)
    df["left_hip_x"] = np.where(ok_left, df["left_hip_x"], np.nan)
    df["left_hip_y"] = np.where(ok_left, df["left_hip_y"], np.nan)

    # Tâm hông = trung bình hai hông, chỉ tính khi cả hai hông đủ tin cậy
    df["hip_mid_y"] = np.where(
        ok_left & ok_right,
        (df["left_hip_y"] + df["right_hip_y"]) / 2,
        np.nan,
    )
    return df


def speed_series(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bỏ NaN, rồi tính chênh lệch giữa hai điểm liên tiếp và tốc độ |dy/dt| (pixel/giây)."""
    mask = ~np.isnan(y)
    t, y = t[mask], y[mask]
    dy = np.diff(y)
    dt = np.diff(t)
    speed = np.abs(dy / dt)
    return t[1:], np.abs(dy), speed


def print_stats(label: str, abs_dy: np.ndarray, speed: np.ndarray) -> None:
    if len(abs_dy) == 0:
        print(f"  {label}: không đủ dữ liệu")
        return
    print(f"  {label}:")
    print(f"    |Δy| giữa 2 điểm liên tiếp: trung vị = {np.median(abs_dy):.2f} px, "
          f"P95 = {np.percentile(abs_dy, 95):.2f} px, max = {abs_dy.max():.2f} px")
    print(f"    Tốc độ |dy/dt|            : trung vị = {np.median(speed):.0f} px/s, "
          f"P95 = {np.percentile(speed, 95):.0f} px/s, max = {speed.max():.0f} px/s")


def plot_one(csv_path: Path, kp_conf_min: float, out_dir: Path) -> None:
    df = load_hip_series(csv_path, kp_conf_min)
    t = df["time_s"].to_numpy()
    name = csv_path.stem

    print(f"\n=== {name} ===")
    print(f"  Dòng của người #0: {len(df)} | hông trái bị loại (conf thấp): "
          f"{df['left_hip_y'].isna().sum()} | tâm hông bị loại: {df['hip_mid_y'].isna().sum()}")

    _t_left, dy_left, sp_left = speed_series(t, df["left_hip_y"].to_numpy())
    t_mid, _dy_mid, sp_mid = speed_series(t, df["hip_mid_y"].to_numpy())
    print_stats("Hông trái (y)", dy_left, sp_left)
    print_stats("Tâm hông  (y)", _dy_mid, sp_mid)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    # (1) Tọa độ y. Đảo trục vì trong ảnh y tăng XUỐNG DƯỚI:
    #     sau khi đảo, đường đi xuống = người hạ thấp xuống, dễ đọc hơn.
    ax1.plot(t, df["left_hip_y"], label="Hông trái", alpha=0.6)
    ax1.plot(t, df["hip_mid_y"], label="Tâm hông", linewidth=1.5)
    ax1.invert_yaxis()
    ax1.set_ylabel("y (px)\n↑ cao hơn")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # (2) Tọa độ x của hông trái — để chẩn đoán độ rung theo chiều ngang ở M1.2
    ax2.plot(t, df["left_hip_x"], color="tab:green", alpha=0.8)
    ax2.set_ylabel("x hông trái (px)")
    ax2.grid(alpha=0.3)

    # (3) Tốc độ dọc của tâm hông giữa hai frame liên tiếp
    ax3.plot(t_mid, sp_mid, color="tab:red", linewidth=1)
    ax3.set_ylabel("|tốc độ y|\n(px/s)")
    ax3.set_xlabel("Thời gian (giây)")
    ax3.grid(alpha=0.3)

    fig.suptitle(f"{name} — hông theo thời gian (kp_conf ≥ {kp_conf_min})")
    fig.tight_layout()
    out_path = out_dir / f"{name}_hip.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Đã lưu biểu đồ: {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for csv in args.csv:
        plot_one(Path(csv), args.kp_conf, out_dir)


if __name__ == "__main__":
    main()
