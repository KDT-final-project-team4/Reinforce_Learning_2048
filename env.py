import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logic

# ==========================================
# 🎯 AI 보상(Reward) 설정 
# (학습 방식을 바꾸고 싶다면 이 숫자들만 수정하세요!)
# ==========================================
PENALTY_INVALID_MOVE = -10  # 변화가 없는 방향(벽에 막힌 곳)으로 움직였을 때 주는 페널티 (음수)
REWARD_VALID_MOVE = 0       # 타일이 합쳐지진 않았지만, 정상적으로 이동했을 때 주는 기본 점수
SCORE_MULTIPLIER = 1.0      # 게임 내 합체 점수(scoreGain)를 AI 보상으로 변환하는 배율
# ==========================================

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        
        # 행동 공간: 0: "up", 1: "down", 2: "left", 3: "right"
        self.action_space = spaces.Discrete(4)
        
        # 상태 공간: 4x4 보드를 1차원 배열(길이 16)로 펼침 (로그 스케일 적용)
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

    def step(self, action):
        directions = {0: "up", 1: "down", 2: "left", 3: "right"}
        dir_str = directions[action]
        
        sim = logic.simulateMove(self.board, dir_str)
        
        reward = 0
        terminated = False
        
        if not sim["changed"]:
            # 보드에 변화가 없는 무의미한 방향 입력 시 페널티 적용
            reward = PENALTY_INVALID_MOVE
        else:
            # 보드 상태 갱신 및 합체 점수 기반 보상 계산
            self.board = sim["result"]
            
            # 여기서 설정한 보상값이 적용됩니다.
            reward = REWARD_VALID_MOVE + (sim["scoreGain"] * SCORE_MULTIPLIER)
            
            logic.addRandomTile(self.board)
            
        if logic.isGameOver(self.board):
            terminated = True
            
        return self._get_obs(), reward, terminated, False, {}