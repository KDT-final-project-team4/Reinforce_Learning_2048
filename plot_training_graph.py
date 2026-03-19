import os
import pandas as pd
import matplotlib.pyplot as plt

import config


def smooth(series, window=50):
    return series.rolling(window=window, min_periods=1).mean()


def plot_single(x, y, title, ylabel, save_name, label=None):
    plt.figure(figsize=(14, 8))

    if label:
        plt.plot(x, y, linewidth=3, label=label)
        plt.legend()
    else:
        plt.plot(x, y, linewidth=3)

    plt.title(title, fontsize=18)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(config.MODEL_DIR, save_name)
    plt.savefig(save_path, dpi=200)
    print(f"✅ 저장 완료: {save_path}")

    plt.show()


def main():
    log_path = os.path.join(config.MODEL_DIR, "train_log.csv")
    df = pd.read_csv(log_path)

    # 숫자 변환
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # 🔥 스무딩 추가
    df["avg_score_s"] = smooth(df["avg_score"], 50)
    df["avg_max_tile_s"] = smooth(df["avg_max_tile"], 50)
    df["rate_1024_s"] = smooth(df["rate_1024"], 50)
    df["rate_2048_s"] = smooth(df["rate_2048"], 50)

    # =========================
    # 1. Average Score
    # =========================
    plot_single(
        df["episode"],
        df["avg_score_s"],
        "Average Score (Smoothed)",
        "Score",
        "avg_score.png"
    )

    # =========================
    # 2. Average Max Tile
    # =========================
    plot_single(
        df["episode"],
        df["avg_max_tile_s"],
        "Average Max Tile (Smoothed)",
        "Tile",
        "avg_max_tile.png"
    )

    # =========================
    # 3. Reach Rate
    # =========================
    plt.figure(figsize=(14, 8))
    plt.plot(df["episode"], df["rate_1024_s"], linewidth=3, label="1024")
    plt.plot(df["episode"], df["rate_2048_s"], linewidth=3, label="2048")

    plt.title("Reach Rate (Smoothed)", fontsize=18)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Rate (%)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(config.MODEL_DIR, "reach_rate.png")
    plt.savefig(save_path, dpi=200)
    print(f"✅ 저장 완료: {save_path}")

    plt.show()

    # =========================
    # 4. Best Average Score
    # =========================
    plot_single(
        df["episode"],
        df["best_avg_score"],
        "Best Average Score",
        "Score",
        "best_avg_score.png"
    )


if __name__ == "__main__":
    main()