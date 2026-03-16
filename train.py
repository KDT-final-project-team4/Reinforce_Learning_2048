import os
import pickle
import numpy as np
import csv

import config
from agent import Agent2048
from env import Game2048Env
from ntuple_network import NTupleNetwork


def save_training_state(net, episode, scores, max_tiles, reached_1024, reached_2048, reached_4096, reached_8192, best_avg_score):
    net.save(config.CHECKPOINT_PATH)

    state = {
        "episode": episode,
        "scores": scores,
        "max_tiles": max_tiles,
        "reached_1024": reached_1024,
        "reached_2048": reached_2048,
        "reached_4096": reached_4096,
        "reached_8192": reached_8192,
        "best_avg_score": best_avg_score,
    }

    with open(config.TRAIN_STATE_PATH, "wb") as f:
        pickle.dump(state, f)


def load_training_state():
    if not os.path.exists(config.TRAIN_STATE_PATH):
        return None

    with open(config.TRAIN_STATE_PATH, "rb") as f:
        return pickle.load(f)


def init_log():
    log_path = os.path.join(config.MODEL_DIR, "train_log.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "avg_score",
                "avg_max_tile",
                "rate_1024",
                "rate_2048",
                "rate_4096",
                "rate_8192",
                "best_avg_score"
            ])


def write_log(episode, avg_score, avg_max_tile, rate_1024, rate_2048, rate_4096, rate_8192, best_avg_score):
    log_path = os.path.join(config.MODEL_DIR, "train_log.csv")

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            avg_score,
            avg_max_tile,
            rate_1024,
            rate_2048,
            rate_4096,
            rate_8192,
            best_avg_score
        ])


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    init_log()

    env = Game2048Env()
    net = NTupleNetwork()
    agent = Agent2048(env, net)

    scores = []
    max_tiles = []
    reached_1024 = []
    reached_2048 = []
    reached_4096 = []
    reached_8192 = []

    start_episode = 0
    best_avg_score = -1.0

    # resume
    if os.path.exists(config.CHECKPOINT_PATH) and os.path.exists(config.TRAIN_STATE_PATH):
        print("기존 체크포인트를 불러와서 이어서 학습합니다.")
        net.load(config.CHECKPOINT_PATH)

        state = load_training_state()
        start_episode = state["episode"] + 1
        scores = state["scores"]
        max_tiles = state["max_tiles"]
        reached_1024 = state["reached_1024"]
        reached_2048 = state["reached_2048"]
        reached_4096 = state.get("reached_4096", [])
        reached_8192 = state.get("reached_8192", [])
        best_avg_score = state.get("best_avg_score", -1.0)

    for episode in range(start_episode, config.NUM_EPISODES):
        board = env.reset()
        done = False

        while not done:
            action, afterstate, _ = agent.select_action(board)

            if action is None or afterstate is None:
                break

            next_state, done = env.step(action)

            # afterstate TD target
            if done:
                target = 0.0
            else:
                next_action, next_afterstate, next_reward = agent.select_action(next_state)

                if next_action is None or next_afterstate is None:
                    target = 0.0
                else:
                    target = next_reward + net.get_value(next_afterstate)

            current_value = net.get_value(afterstate)
            td_error = target - current_value
            net.update(afterstate, td_error)

            board = next_state

        final_score = env.score
        max_tile = int(np.max(board))

        scores.append(final_score)
        max_tiles.append(max_tile)
        reached_1024.append(1 if max_tile >= 1024 else 0)
        reached_2048.append(1 if max_tile >= 2048 else 0)
        reached_4096.append(1 if max_tile >= 4096 else 0)
        reached_8192.append(1 if max_tile >= 8192 else 0)

        if episode % 100 == 0:
            recent_scores = scores[-100:]
            recent_tiles = max_tiles[-100:]
            recent_1024 = reached_1024[-100:]
            recent_2048 = reached_2048[-100:]
            recent_4096 = reached_4096[-100:]
            recent_8192 = reached_8192[-100:]

            avg_score = float(np.mean(recent_scores))
            avg_max_tile = float(np.mean(recent_tiles))
            rate_1024 = float(np.mean(recent_1024) * 100)
            rate_2048 = float(np.mean(recent_2048) * 100)
            rate_4096 = float(np.mean(recent_4096) * 100)
            rate_8192 = float(np.mean(recent_8192) * 100)

            print(
                f"{episode} "
                f"avg score {avg_score:.2f} | "
                f"avg max tile {avg_max_tile:.2f} | "
                f"1024 rate {rate_1024:.1f}% | "
                f"2048 rate {rate_2048:.1f}% | "
                f"4096 rate {rate_4096:.1f}% | "
                f"8192 rate {rate_8192:.1f}%"
            )

            write_log(
                episode,
                avg_score,
                avg_max_tile,
                rate_1024,
                rate_2048,
                rate_4096,
                rate_8192,
                best_avg_score
            )

        # best model 저장
        if len(scores) >= config.EVAL_WINDOW:
            current_avg = float(np.mean(scores[-config.EVAL_WINDOW:]))
            if current_avg > best_avg_score:
                best_avg_score = current_avg
                net.save(config.BEST_MODEL_PATH)
                print(
                    f"[BEST] episode {episode} | "
                    f"avg score ({config.EVAL_WINDOW}) = {best_avg_score:.2f}"
                )

        # periodic checkpoint
        if episode % config.CHECKPOINT_EVERY == 0 and episode > 0:
            save_training_state(
                net,
                episode,
                scores,
                max_tiles,
                reached_1024,
                reached_2048,
                reached_4096,
                reached_8192,
                best_avg_score,
            )
            print(f"[CHECKPOINT] saved at episode {episode}")

    # final save
    net.save(config.MODEL_PATH)

    save_training_state(
        net,
        config.NUM_EPISODES - 1,
        scores,
        max_tiles,
        reached_1024,
        reached_2048,
        reached_4096,
        reached_8192,
        best_avg_score,
    )

    print(f"final model saved -> {config.MODEL_PATH}")
    print(f"best model saved -> {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()