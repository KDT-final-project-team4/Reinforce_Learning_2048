import pygame
import numpy as np
import logic
import colors
import os
import pickle
import math

# --- 게임 기본 설정 ---
gameWidth = 600
gameHeight = 700
totalWidth = gameWidth * 2 

tileSize = 120
gap = 10
startY = 150

startX_AI = (gameWidth - (tileSize * 4 + gap * 3)) / 2
startX_Player = gameWidth + (gameWidth - (tileSize * 4 + gap * 3)) / 2

BG_COLOR = (250, 248, 239)
BOARD_BG = (187, 173, 160)
TEXT_COLOR = (119, 110, 101)
OVER_TEXT_COLOR = (255, 0, 0)

# --- N-Tuple Network 모델 추론 클래스 ---
class NTupleAgent:
    def __init__(self, model_path="2048_ntuple_model.pkl"):
        self.lut = {}
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.lut = pickle.load(f)
            print("🚀 N-Tuple 모델을 성공적으로 불러왔습니다!")
        else:
            print("모델 파일을 찾을 수 없어 랜덤으로 동작합니다.")

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
        if not self.lut:
            return 0.0
        tuples = self._get_tuples(board)
        return sum(self.lut.get(t, 0.0) for t in tuples)

    def predict(self, board):
        directions = ["up", "down", "left", "right"]
        best_action = None
        best_value = -float('inf')
        
        # 각 방향으로 움직였을 때(After-state)의 가치를 평가하여 최적의 수 선택
        for dir_str in directions:
            sim = logic.simulateMove(board, dir_str)
            if sim["changed"]:
                v = sim["scoreGain"] + self.get_value(sim["result"])
                if v > best_value:
                    best_value = v
                    best_action = dir_str
        
        # 만약 모델이 비어있거나 결정하지 못했다면 랜덤 선택
        if best_action is None:
            best_action = directions[np.random.randint(4)]
            
        return best_action

def draw_board(screen, font, title_font, board, startX, score, gameOver, title_text):
    title_surface = title_font.render(title_text, True, TEXT_COLOR)
    title_rect = title_surface.get_rect(center=(startX + (tileSize*4 + gap*3)/2, 50))
    screen.blit(title_surface, title_rect)
    
    score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
    score_rect = score_surface.get_rect(center=(startX + (tileSize*4 + gap*3)/2, 100))
    screen.blit(score_surface, score_rect)

    board_w = tileSize * 4 + gap * 5
    board_h = tileSize * 4 + gap * 5
    pygame.draw.rect(screen, BOARD_BG, (startX - gap, startY - gap, board_w, board_h), border_radius=6)

    for i in range(4):
        for j in range(4):
            val = board[i][j]
            x = startX + j * (tileSize + gap)
            y = startY + i * (tileSize + gap)
            
            bg_color = colors.getTileBg(val)
            pygame.draw.rect(screen, bg_color, (x, y, tileSize, tileSize), border_radius=3)
            
            if val > 0:
                fg_color = colors.getTileFg(val)
                font_size = 32 if val >= 1024 else 36 if val >= 128 else 42
                temp_font = pygame.font.SysFont("arial", font_size, bold=True)
                val_surface = temp_font.render(str(val), True, fg_color)
                val_rect = val_surface.get_rect(center=(x + tileSize/2, y + tileSize/2))
                screen.blit(val_surface, val_rect)

    if gameOver:
        over_surface = title_font.render("GAME OVER", True, OVER_TEXT_COLOR)
        over_rect = over_surface.get_rect(center=(startX + (tileSize*4 + gap*3)/2, startY + board_h + 40))
        screen.blit(over_surface, over_rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((totalWidth, gameHeight))
    pygame.display.set_caption("2048: AI (N-Tuple) vs Player")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 28, bold=True)
    title_font = pygame.font.SysFont("arial", 48, bold=True)

    ai_board = logic.makeEmptyBoard()
    logic.addRandomTile(ai_board)
    logic.addRandomTile(ai_board)
    ai_score = 0
    ai_gameOver = False

    player_board = logic.makeEmptyBoard()
    logic.addRandomTile(player_board)
    logic.addRandomTile(player_board)
    player_score = 0
    player_gameOver = False

    agent = NTupleAgent()

    ai_move_timer = 0
    ai_move_delay = 50  

    mouse_start_x = 0
    mouse_start_y = 0
    is_dragging = False

    running = True
    while running:
        dt = clock.tick(60) 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not player_gameOver:
                dir_str = None
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: dir_str = "up"
                    elif event.key == pygame.K_DOWN: dir_str = "down"
                    elif event.key == pygame.K_LEFT: dir_str = "left"
                    elif event.key == pygame.K_RIGHT: dir_str = "right"
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_start_x, mouse_start_y = event.pos
                    is_dragging = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if is_dragging:
                        is_dragging = False
                        mouse_end_x, mouse_end_y = event.pos
                        dx = mouse_end_x - mouse_start_x
                        dy = mouse_end_y - mouse_start_y
                        if abs(dx) > 20 or abs(dy) > 20:
                            if abs(dx) > abs(dy): dir_str = "right" if dx > 0 else "left"
                            else: dir_str = "down" if dy > 0 else "up"

                if dir_str:
                    sim = logic.simulateMove(player_board, dir_str)
                    if sim["changed"]:
                        player_board = sim["result"]
                        player_score += sim["scoreGain"]
                        logic.addRandomTile(player_board)
                        if logic.isGameOver(player_board):
                            player_gameOver = True

        if not ai_gameOver:
            ai_move_timer += dt
            if ai_move_timer > ai_move_delay:
                ai_move_timer = 0
                
                # N-Tuple 모델 예측 호출
                ai_dir_str = agent.predict(ai_board)
                sim = logic.simulateMove(ai_board, ai_dir_str)
                
                if sim["changed"]:
                    ai_board = sim["result"]
                    ai_score += sim["scoreGain"]
                    logic.addRandomTile(ai_board)
                    if logic.isGameOver(ai_board):
                        ai_gameOver = True
                else:
                    # 안전 장치
                    fallback_dir = ["up", "down", "left", "right"][np.random.randint(4)]
                    sim_fallback = logic.simulateMove(ai_board, fallback_dir)
                    if sim_fallback["changed"]:
                        ai_board = sim_fallback["result"]
                        ai_score += sim_fallback["scoreGain"]
                        logic.addRandomTile(ai_board)
                        if logic.isGameOver(ai_board):
                            ai_gameOver = True

        screen.fill(BG_COLOR)
        pygame.draw.line(screen, (200, 200, 200), (gameWidth, 0), (gameWidth, gameHeight), 2)
        draw_board(screen, font, title_font, ai_board, startX_AI, ai_score, ai_gameOver, "AI (N-Tuple)")
        draw_board(screen, font, title_font, player_board, startX_Player, player_score, player_gameOver, "Player")

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()