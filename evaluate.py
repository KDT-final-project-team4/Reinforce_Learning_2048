import numpy as np

import config
from agent import Agent2048
from env import Game2048Env
from ntuple_network import NTupleNetwork


def main():
    env = Game2048Env()
    net = NTupleNetwork()

    model_path = config.BEST_MODEL_PATH if config.USE_EXPECTIMAX_IN_GAME else config.MODEL_PATH
    net.load(model_path)

    agent = Agent2048(env, net)

    scores = []
    max_tiles = []

    num_games = 100

    for _ in range(num_games):
        board = env.reset()
        done = False

        while not done:
            if config.USE_EXPECTIMAX_IN_GAME:
                action = agent.expectimax_action(board, depth=config.EXPECTIMAX_DEPTH)
            else:
                action, _, _ = agent.select_action(board)

            if action is None:
                break

            board, done = env.step(action)

        scores.append(env.score)
        max_tiles.append(int(np.max(board)))

    print(f"games: {num_games}")
    print(f"avg score: {np.mean(scores):.2f}")
    print(f"avg max tile: {np.mean(max_tiles):.2f}")
    print(f"best score: {np.max(scores)}")
    print(f"best tile: {np.max(max_tiles)}")
    print(f"2048 rate: {np.mean(np.array(max_tiles) >= 2048) * 100:.1f}%")


if __name__ == "__main__":
    main()