import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logic
import math

# ==========================================
# 🎯 AI 보상(Reward) 및 페널티 설정 
# (학습 방식을 바꾸고 싶다면 이 숫자들만 수정하세요!)
# ==========================================
PENALTY_INVALID_MOVE = -10.0       # 변화가 없는 방향(벽에 막힌 곳)으로 움직였을 때 주는 페널티
REWARD_VALID_MOVE = 0.0            # 타일이 합쳐지진 않았지만, 정상적으로 이동했을 때 주는 기본 점수
SCORE_MULTIPLIER = 1.0             # 게임 내 합체 점수(scoreGain)를 AI 보상으로 변환하는 배율
# ==========================================

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=16, shape=(16,), dtype=np.float32)
        self.board = logic.makeEmptyBoard()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = logic.makeEmptyBoard()
        logic.addRandomTile(self.board)
        logic.addRandomTile(self.board)
        return self._get_obs(), {}
        
    def _get_obs(self):
        obs = np.zeros(16, dtype=np.float32)
        idx = 0
        for i in range(4):
            for j in range(4):
                val = self.board[i][j]
                if val > 0:
                    obs[idx] = np.log2(val)
                idx += 1
        return obs

    def get_board(self):
        return self.board

    # 🔥 N-Tuple 학습을 위해 갈 수 있는 모든 방향의 시뮬레이션 결과를 반환
    def get_simulated_moves(self):
        directions = ["up", "down", "left", "right"]
        valid_moves = {}
        for dir_str in directions:
            sim = logic.simulateMove(self.board, dir_str)
            if sim["changed"]:
                valid_moves[dir_str] = sim
        return valid_moves

    # 🔥 AI가 선택한 최고의 시뮬레이션 결과를 실제 보드에 적용하고 턴 넘기기
    def step_after_state(self, best_sim):
        self.board = best_sim["result"]
        
        # 🚨 상단에서 설정한 커스텀 보상값이 적용됩니다.
        reward = REWARD_VALID_MOVE + (best_sim["scoreGain"] * SCORE_MULTIPLIER)
        
        logic.addRandomTile(self.board)
        
        terminated = logic.isGameOver(self.board)
        max_tile = max(max(r) for r in self.board)
        
        return self._get_obs(), reward, terminated, False, {"highest": max_tile, "scoreGain": reward}

    # (기존 방식 유지) 일반적인 스텝 함수
    def step(self, action):
        directions = {0: "up", 1: "down", 2: "left", 3: "right"}
        dir_str = directions[action]
        
        sim = logic.simulateMove(self.board, dir_str)
        reward = 0.0
        terminated = False
        info = {"scoreGain": 0, "highest": max(max(r) for r in self.board)}
        
        if not sim["changed"]:
            # 🚨 상단에서 설정한 페널티가 적용됩니다.
            reward = PENALTY_INVALID_MOVE
        else:
            self.board = sim["result"]
            
            # 🚨 상단에서 설정한 커스텀 보상값이 적용됩니다.
            reward = REWARD_VALID_MOVE + (sim["scoreGain"] * SCORE_MULTIPLIER)
            
            info["scoreGain"] = sim["scoreGain"]
            logic.addRandomTile(self.board)
            info["highest"] = max(max(r) for r in self.board)
            
        if logic.isGameOver(self.board):
            terminated = True
            
        return self._get_obs(), reward, terminated, False, info