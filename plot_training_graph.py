import os
import pandas as pd
import matplotlib.pyplot as plt

import config


def main():
    log_path = os.path.join(config.MODEL_DIR, "train_log.csv")

    if not os.path.exists(log_path):
        print(f"로그 파일이 없습니다: {log_path}")
        return

    df = pd.read_csv(log_path)

    if df.empty:
        print("로그 파일이 비어 있습니다.")
        return

    required_columns = [
        "episode",
        "avg_score",
        "avg_max_tile",
        "rate_1024",
        "rate_2048",
        "best_avg_score",
    ]

    for col in required_columns:
        if col not in df.columns:
            print(f"필수 컬럼이 없습니다: {col}")
            return

    plt.figure(figsize=(14, 10))

    # 1. Average Score
    plt.subplot(2, 2, 1)
    plt.plot(df["episode"], df["avg_score"], marker="o", linewidth=2)
    plt.title("Average Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)

    # 2. Average Max Tile
    plt.subplot(2, 2, 2)
    plt.plot(df["episode"], df["avg_max_tile"], marker="o", linewidth=2)
    plt.title("Average Max Tile")
    plt.xlabel("Episode")
    plt.ylabel("Tile")
    plt.grid(True, alpha=0.3)

    # 3. 1024 / 2048 Rate
    plt.subplot(2, 2, 3)
    plt.plot(df["episode"], df["rate_1024"], marker="o", linewidth=2, label="1024 Rate")
    plt.plot(df["episode"], df["rate_2048"], marker="o", linewidth=2, label="2048 Rate")
    plt.title("Reach Rate")
    plt.xlabel("Episode")
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Best Average Score
    plt.subplot(2, 2, 4)
    plt.plot(df["episode"], df["best_avg_score"], marker="o", linewidth=2)
    plt.title("Best Average Score")
    plt.xlabel("Episode")
    plt.ylabel("Best Avg Score")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(config.MODEL_DIR, "training_graph.png")
    plt.savefig(save_path, dpi=150)
    print(f"그래프 저장 완료: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()