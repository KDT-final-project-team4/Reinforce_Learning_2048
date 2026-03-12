import copy

import config
from reward import reward_function


class Agent2048:
    def __init__(self, env, net):
        self.env = env
        self.net = net

    def select_action(self, board):
        """
        Training/eval default:
        choose action maximizing reward(after move) + V(afterstate)
        """
        best_action = None
        best_afterstate = None
        best_reward = 0.0
        best_value = -1e18

        for action in range(4):
            sim = self.env.simulate_move(board, action)
            if not sim["valid"]:
                continue

            afterstate = sim["result"]
            merge_score = sim["scoreGain"]
            r = reward_function(afterstate, merge_score, done=False)
            v = r + self.net.get_value(afterstate)

            if v > best_value:
                best_value = v
                best_action = action
                best_afterstate = afterstate
                best_reward = r

        return best_action, best_afterstate, best_reward

    def expectimax_action(self, board, depth=None):
        depth = config.EXPECTIMAX_DEPTH if depth is None else depth

        best_action = None
        best_value = -1e18

        for action in range(4):
            sim = self.env.simulate_move(board, action)
            if not sim["valid"]:
                continue

            afterstate = sim["result"]
            merge_score = sim["scoreGain"]
            immediate = reward_function(afterstate, merge_score, done=False)

            value = immediate + self._chance_value(afterstate, depth)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _chance_value(self, afterstate, depth):
        successors = self.env.get_random_successors(afterstate)

        if not successors:
            return 0.0

        expected = 0.0
        for next_state, prob in successors:
            if depth <= 0:
                expected += prob * self.net.get_value(next_state)
            else:
                expected += prob * self._max_value(next_state, depth - 1)
        return expected

    def _max_value(self, state, depth):
        best = -1e18
        has_valid = False

        for action in range(4):
            sim = self.env.simulate_move(state, action)
            if not sim["valid"]:
                continue

            has_valid = True
            afterstate = sim["result"]
            merge_score = sim["scoreGain"]
            immediate = reward_function(afterstate, merge_score, done=False)

            if depth <= 0:
                v = immediate + self.net.get_value(afterstate)
            else:
                v = immediate + self._chance_value(afterstate, depth)

            if v > best:
                best = v

        return 0.0 if not has_valid else best