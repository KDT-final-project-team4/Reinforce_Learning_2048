import random
import numpy as np


class Game2048Env:
    def __init__(self):
        self.size = 4
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.last_move_score_gain = 0

    # -------------------------------------------------
    # Core helpers
    # -------------------------------------------------
    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.last_move_score_gain = 0
        self._add_random_tile(self.board)
        self._add_random_tile(self.board)
        return self.board.copy()

    def _add_random_tile(self, board):
        empties = list(zip(*np.where(board == 0)))
        if not empties:
            return False

        r, c = random.choice(empties)
        board[r][c] = 2 if random.random() < 0.9 else 4
        return True

    def can_move(self, board=None):
        board = self.board if board is None else board

        if np.any(board == 0):
            return True

        for r in range(4):
            for c in range(3):
                if board[r][c] == board[r][c + 1]:
                    return True

        for c in range(4):
            for r in range(3):
                if board[r][c] == board[r + 1][c]:
                    return True

        return False

    # -------------------------------------------------
    # Movement logic
    # action: 0 up, 1 down, 2 left, 3 right
    # -------------------------------------------------
    def move_left(self, board):
        new_board = np.zeros_like(board)
        moved = False
        score_gain = 0

        for r in range(4):
            row = [v for v in board[r] if v != 0]
            merged = []
            i = 0

            while i < len(row):
                if i + 1 < len(row) and row[i] == row[i + 1]:
                    v = row[i] * 2
                    merged.append(v)
                    score_gain += v
                    i += 2
                else:
                    merged.append(row[i])
                    i += 1

            merged += [0] * (4 - len(merged))
            new_board[r] = merged

            if not np.array_equal(board[r], new_board[r]):
                moved = True

        return new_board, score_gain, moved

    def move_right(self, board):
        flipped = np.fliplr(board)
        moved_board, score_gain, moved = self.move_left(flipped)
        return np.fliplr(moved_board), score_gain, moved

    def move_up(self, board):
        rotated = np.rot90(board, 1)
        moved_board, score_gain, moved = self.move_left(rotated)
        return np.rot90(moved_board, -1), score_gain, moved

    def move_down(self, board):
        rotated = np.rot90(board, -1)
        moved_board, score_gain, moved = self.move_left(rotated)
        return np.rot90(moved_board, 1), score_gain, moved

    def simulate_move(self, board, action):
        temp = np.array(board, dtype=int).copy()

        if action == 0:
            result, score_gain, valid = self.move_up(temp)
        elif action == 1:
            result, score_gain, valid = self.move_down(temp)
        elif action == 2:
            result, score_gain, valid = self.move_left(temp)
        elif action == 3:
            result, score_gain, valid = self.move_right(temp)
        else:
            return {"valid": False, "result": temp, "scoreGain": 0}

        return {
            "valid": valid,
            "result": result,
            "scoreGain": score_gain,
        }

    def step(self, action):
        sim = self.simulate_move(self.board, action)

        if not sim["valid"]:
            done = not self.can_move(self.board)
            self.last_move_score_gain = 0
            return self.board.copy(), done

        self.board = sim["result"]
        self.score += sim["scoreGain"]
        self.last_move_score_gain = sim["scoreGain"]

        self._add_random_tile(self.board)
        done = not self.can_move(self.board)
        return self.board.copy(), done

    # -------------------------------------------------
    # Chance node successors for expectimax
    # -------------------------------------------------
    def get_random_successors(self, afterstate):
        afterstate = np.array(afterstate, dtype=int)
        empties = list(zip(*np.where(afterstate == 0)))
        if not empties:
            return []

        successors = []
        p_cell = 1.0 / len(empties)

        for (r, c) in empties:
            b2 = afterstate.copy()
            b2[r][c] = 2
            successors.append((b2, p_cell * 0.9))

            b4 = afterstate.copy()
            b4[r][c] = 4
            successors.append((b4, p_cell * 0.1))

        return successors