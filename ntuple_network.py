import math
import os
import pickle
from collections import defaultdict

import config
from symmetry import get_symmetries


def encode_tile(value: int) -> int:
    if value == 0:
        return 0
    return int(math.log2(value))


class NTupleNetwork:
    def __init__(self, alpha=None, tuples=None):
        self.alpha = config.ALPHA if alpha is None else alpha
        self.tuples = config.TUPLES if tuples is None else tuples

        self.weights = defaultdict(float)
        self.visit_counts = defaultdict(int)

    def tuple_index(self, board, pattern):
        return tuple(encode_tile(int(board[r][c])) for r, c in pattern)

    def get_value(self, board):
        boards = get_symmetries(board)
        total = 0.0

        for b in boards:
            for pattern in self.tuples:
                idx = self.tuple_index(b, pattern)
                total += self.weights[idx]

        # symmetry와 tuple 개수 둘 다 반영해서 scale 안정화
        return total / (len(boards) * len(self.tuples))

    def update(self, board, td_error):
        boards = get_symmetries(board)
        feature_count = len(boards) * len(self.tuples)

        if config.USE_OPTIMISTIC_TD:
            scale = config.POSITIVE_TD_SCALE if td_error >= 0 else config.NEGATIVE_TD_SCALE
            td_error *= scale

        for b in boards:
            for pattern in self.tuples:
                idx = self.tuple_index(b, pattern)
                self.visit_counts[idx] += 1

                if config.USE_VISIT_COUNT_ALPHA:
                    local_alpha = self.alpha / math.sqrt(self.visit_counts[idx])
                    local_alpha = max(local_alpha, config.MIN_LOCAL_ALPHA)
                else:
                    local_alpha = self.alpha

                self.weights[idx] += (local_alpha * td_error) / feature_count

    def state_dict(self):
        return {
            "weights": dict(self.weights),
            "visit_counts": dict(self.visit_counts),
        }

    def load_state_dict(self, state):
        self.weights = defaultdict(float, state.get("weights", {}))
        self.visit_counts = defaultdict(int, state.get("visit_counts", {}))

    def save(self, path=None):
        path = config.MODEL_PATH if path is None else path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path=None):
        path = config.MODEL_PATH if path is None else path

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.load_state_dict(state)