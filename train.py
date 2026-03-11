import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from env import Game2048Env
import numpy as np

class Print100EpisodesCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        
        # 1판 상태
        self.current_score = 0
        self.current_steps = 0
        
        # 100판 기록용 리스트
        self.scores_window = []
        self.steps_window = []
        self.max_tiles_window = []
        
        # FPS와 시간 측정을 위한 변수
        self.start_time = None

    def _on_training_start(self) -> None:
        # 학습 시작 시간 기록
        self.start_time = time.time()

    def _on_step(self) -> bool:
        self.current_steps += 1
        reward = self.locals['rewards'][0]
        
        if reward > 0:
            self.current_score += reward
        
        done = self.locals['dones'][0]
        if done:
            self.episode_count += 1
            
            self.scores_window.append(self.current_score)
            self.steps_window.append(self.current_steps)

            # 마지막 보드 상태에서 가장 큰 타일 계산
            info = self.locals['infos'][0]
            max_tile = 0
            if 'terminal_observation' in info:
                terminal_obs = info['terminal_observation']
                max_log = np.max(terminal_obs)
                if max_log > 0:
                    max_tile = int(2 ** max_log)
            self.max_tiles_window.append(max_tile)
            
            # 100판마다 한글 표 출력
            if self.episode_count % 100 == 0:
                avg_score = sum(self.scores_window) / len(self.scores_window)
                avg_steps = sum(self.steps_window) / len(self.steps_window)
                max_tile_ever = max(self.max_tiles_window)
                avg_max_tile = sum(self.max_tiles_window) / len(self.max_tiles_window)
                
                # 시간 및 프레임 계산
                time_elapsed = int(time.time() - self.start_time)
                fps = int(self.num_timesteps / time_elapsed) if time_elapsed > 0 else 0
                
                # DQN 모델 내부 변수 가져오기 (탐험 확률, 업데이트 횟수)
                exp_rate = getattr(self.model, 'exploration_rate', 0.0)
                n_updates = getattr(self.model, '_n_updates', 0)
                
                # 로거에서 loss 값 가져오기 (없으면 0.0)
                loss = 0.0
                if hasattr(self.model, 'logger') and self.model.logger:
                    loss = self.model.logger.name_to_value.get('train/loss', 0.0)
                
                # 영어 표를 완벽하게 대체하는 한글 표
                print(f"\n----------------------------------")
                print(f"| 진행 상황 (rollout)   |          |")
                print(f"|    평균 이동 횟수     | {avg_steps:<8.1f} |")
                print(f"|    평균 획득 점수     | {avg_score:<8.1f} |")
                print(f"|    탐험 확률 (랜덤)   | {exp_rate:<8.3f} |")
                print(f"| 시간 (time)           |          |")
                print(f"|    진행된 판 수       | {self.episode_count:<8} |")
                print(f"|    초당 스텝 (fps)    | {fps:<8} |")
                print(f"|    경과 시간 (초)     | {time_elapsed:<8} |")
                print(f"|    총 누적 스텝       | {self.num_timesteps:<8} |")
                print(f"| 학습 (train)          |          |")
                print(f"|    손실값 (loss)      | {loss:<8.4f} |")
                print(f"|    업데이트 횟수      | {n_updates:<8} |")
                print(f"----------------------------------")
                print(f" 🏆 [100판 요약] 최고 타일: {int(max_tile_ever)} | 평균 타일: {avg_max_tile:.1f}")
                print(f"----------------------------------\n")
                
                # 다음 100판 기록을 위해 초기화
                self.scores_window = []
                self.steps_window = []
                self.max_tiles_window = []
                
            self.current_score = 0
            self.current_steps = 0
            
        return True

def main():
    env = Game2048Env()
    
    # verbose=0 으로 설정하여 원래 나오던 영어 표를 완전히 차단합니다.
    model = DQN("MlpPolicy", env, verbose=0)
    
    print("AI 학습을 시작합니다. 100판마다 100% 한글로 번역된 결과 표가 출력됩니다...")
    
    callback = Print100EpisodesCallback()
    
    model.learn(total_timesteps=500000, callback=callback)
    
    model.save("2048_ai_model")
    print("학습 완료! '2048_ai_model.zip' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()