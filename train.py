import time
import pickle
import math
import random
import os
from env import Game2048Env

class NTupleNetwork:
    def __init__(self):
        self.lut = {}
        # 🔥 학습률을 낮춰서 수치가 폭발하는 것을 1차 방지합니다.
        self.alpha = 0.001  

    def _get_tuples(self, board):
        b = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                if board[i][j] > 0:
                    b[i][j] = int(math.log2(board[i][j]))
        
        tuples = []
        for i in range(4):
            tuples.append(f"R{i}_{b[i][0]}_{b[i][1]}_{b[i][2]}_{b[i][3]}")
            tuples.append(f"C{i}_{b[0][i]}_{b[1][i]}_{b[2][i]}_{b[3][i]}")
        for i in range(3):
            for j in range(2):
                tuples.append(f"B23_{i}_{j}_{b[i][j]}_{b[i][j+1]}_{b[i][j+2]}_{b[i+1][j]}_{b[i+1][j+1]}_{b[i+1][j+2]}")
                tuples.append(f"B32_{i}_{j}_{b[j][i]}_{b[j+1][i]}_{b[j+2][i]}_{b[j][i+1]}_{b[j+1][i+1]}_{b[j+2][i+1]}")
        return tuples

    def get_value(self, board):
        tuples = self._get_tuples(board)
        val = sum(self.lut.get(t, 0.0) for t in tuples)
        # 🔥 만약 수치가 꼬였더라도 에러를 내뿜지 않고 0으로 방어합니다.
        if math.isnan(val) or math.isinf(val):
            return 0.0 
        return val

    def update(self, board, error):
        if math.isnan(error) or math.isinf(error):
            return
        
        # 🔥 핵심 안전장치 (Gradient Clipping): 한 번에 너무 큰 깨달음을 얻어 뇌정지가 오는 걸 막습니다.
        error = max(-100.0, min(100.0, error)) 
        
        tuples = self._get_tuples(board)
        for t in tuples:
            self.lut[t] = self.lut.get(t, 0.0) + self.alpha * error

def main():
    # 🚨 이전 학습에서 NaN으로 오염된(뇌정지 온) 파일을 깨끗하게 지우고 새 출발합니다!
    if os.path.exists("2048_ntuple_model.pkl"):
        os.remove("2048_ntuple_model.pkl")
        print("⚠️ 기존에 오염된 모델 파일을 삭제하고 초기화했습니다!")

    env = Game2048Env()
    net = NTupleNetwork()
    
    episodes = 50000 
    
    # 🎲 탐험률(Epsilon) 설정
    epsilon = 0.1         # 초기 탐험 확률 (10% 확률로 엉뚱한 길 가보기)
    epsilon_min = 0.001   # 최소 탐험 확률 (0.1% 유지)
    epsilon_decay = 0.9995 # 매 판마다 서서히 탐험을 줄이고 실력 발휘
    
    print("🧠 [N-Tuple Network + 탐험률 + 안전장치] 안정적인 학습을 시작합니다!")
    
    start_time = time.time()
    
    # 기록용 리스트
    scores_window = []
    steps_window = []
    max_tiles_window = []
    td_errors_window = []
    global_step = 0
    
    for episode in range(1, episodes + 1):
        env.reset()
        score = 0
        steps = 0
        total_td_error = 0.0
        
        while True:
            valid_moves = env.get_simulated_moves()
            
            # 더 이상 움직일 수 없으면 종료 (게임 오버 시 가치 0으로 업데이트)
            if not valid_moves:
                net.update(env.get_board(), 0 - net.get_value(env.get_board()))
                break
            
            best_action = None
            best_sim = None
            
            # 🔥 탐험(Exploration) vs 활용(Exploitation) 적용
            if random.random() < epsilon:
                # 탐험: 갈 수 있는 길 중 아무거나 무작위 선택
                best_action = random.choice(list(valid_moves.keys()))
                best_sim = valid_moves[best_action]
            else:
                # 활용: 머리를 써서 가장 가치가 높은 최고의 방향 선택
                best_value = -float('inf')
                for dir_str, sim in valid_moves.items():
                    v = sim["scoreGain"] + net.get_value(sim["result"])
                    if v > best_value:
                        best_value = v
                        best_action = dir_str
                        best_sim = sim
            
            # 안전장치 (원래는 발생 안 함)
            if best_action is None:
                break
            
            # TD(0) 가치 업데이트 및 오차(Loss) 계산
            current_value = net.get_value(env.get_board())
            target_value = best_sim["scoreGain"] + net.get_value(best_sim["result"])
            
            error = target_value - current_value
            net.update(env.get_board(), error)
            
            # 통계 기록
            total_td_error += abs(error)
            steps += 1
            global_step += 1
            
            # env.py로 실제 턴 진행
            _, reward, terminated, _, info = env.step_after_state(best_sim)
            score += reward
            
            if terminated:
                break
                
        # 한 판이 끝나면 탐험률(Epsilon)을 서서히 낮춰줌
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        scores_window.append(score)
        steps_window.append(steps)
        max_tiles_window.append(info["highest"])
        td_errors_window.append(total_td_error / steps if steps > 0 else 0)
        
        # 🔥 DQN 때처럼 예쁜 100판 요약 표 출력
        if episode % 100 == 0:
            avg_score = sum(scores_window) / len(scores_window)
            avg_steps = sum(steps_window) / len(steps_window)
            avg_max_tile = sum(max_tiles_window) / len(max_tiles_window)
            max_tile_ever = max(max_tiles_window)
            avg_loss = sum(td_errors_window) / len(td_errors_window)
            time_elapsed = int(time.time() - start_time)
            fps = int(global_step / time_elapsed) if time_elapsed > 0 else 0
            
            print(f"\n----------------------------------")
            print(f"| 진행 상황 (rollout)   |          |")
            print(f"|    평균 이동 횟수     | {avg_steps:<8.1f} |")
            print(f"|    평균 획득 점수     | {avg_score:<8.1f} |")
            print(f"|    탐험 확률 (랜덤)   | {epsilon:<8.4f} |")
            print(f"| 시간 (time)           |          |")
            print(f"|    진행된 판 수       | {episode:<8} |")
            print(f"|    초당 스텝 (fps)    | {fps:<8} |")
            print(f"|    경과 시간 (초)     | {time_elapsed:<8} |")
            print(f"|    총 누적 스텝       | {global_step:<8} |")
            print(f"| 학습 (train)          |          |")
            print(f"|    평균 손실값 (Loss) | {avg_loss:<8.4f} |")
            print(f"|    학습된 패턴 수     | {len(net.lut):<8} |")
            print(f"----------------------------------")
            print(f" 🏆 [100판 요약] 이번 최고 타일: {int(max_tile_ever)} | 평균 타일: {avg_max_tile:.1f}")
            print(f"----------------------------------")
            
            scores_window = []
            steps_window = []
            max_tiles_window = []
            td_errors_window = []
            
        # 1만 판마다 모델 세이브
        if episode % 10000 == 0:
            with open("2048_ntuple_model.pkl", "wb") as f:
                pickle.dump(net.lut, f)
            
    with open("2048_ntuple_model.pkl", "wb") as f:
        pickle.dump(net.lut, f)
    print("학습 완료! '2048_ntuple_model.pkl' 파일이 저장되었습니다.")

if __name__ == "__main__":
    main()