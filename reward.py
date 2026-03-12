import numpy as np
import config


def count_empty(board):
    board = np.asarray(board)
    return int(np.sum(board == 0))


def reward_function(board, merge_score, done=False):
    """
    보상은 score를 높이기 위한 도구.
    기본은 실제 2048 score와 정렬되도록 merge_score 중심.
    """
    reward = merge_score * config.REWARD_MERGE_MULT

    if config.REWARD_EMPTY_BONUS != 0.0:
        reward += count_empty(board) * config.REWARD_EMPTY_BONUS

    if done and config.REWARD_LOSE_PENALTY != 0.0:
        reward += config.REWARD_LOSE_PENALTY

    return float(reward)