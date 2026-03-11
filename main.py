import pygame
import numpy as np
import logic
import colors
from stable_baselines3 import DQN
import os

# --- 게임 기본 설정 (기존 main.mjs 값 유지 및 분할화면용 확장) ---
gameWidth = 600
gameHeight = 700
totalWidth = gameWidth * 2  # 좌(AI) 우(Player) 분할

tileSize = 120
gap = 10
startY = 150

# AI 보드 시작 X 좌표 (왼쪽)
startX_AI = (gameWidth - (tileSize * 4 + gap * 3)) / 2
# 플레이어 보드 시작 X 좌표 (오른쪽)
startX_Player = gameWidth + (gameWidth - (tileSize * 4 + gap * 3)) / 2

# 색상
BG_COLOR = (250, 248, 239)      # 0xfaf8ef
BOARD_BG = (187, 173, 160)      # 0xbbada0
CELL_BG = (238, 228, 218)       # 0xeee4da
TEXT_COLOR = (119, 110, 101)    # 0x776e65
OVER_TEXT_COLOR = (255, 0, 0)   # 0xf00

def get_obs(board):
    # env.py와 동일하게 AI 모델에 넣기 위한 상태 변환 함수
    obs = np.zeros(16, dtype=np.float32)
    idx = 0
    for i in range(4):
        for j in range(4):
            val = board[i][j]
            if val > 0:
                obs[idx] = np.log2(val)
            idx += 1
    return obs

def draw_board(screen, font, title_font, board, startX, score, gameOver, title_text):
    # 1. 타이틀 & 점수
    title_surface = title_font.render(title_text, True, TEXT_COLOR)
    title_rect = title_surface.get_rect(center=(startX + (tileSize*4 + gap*3)/2, 50))
    screen.blit(title_surface, title_rect)
    
    score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
    score_rect = score_surface.get_rect(center=(startX + (tileSize*4 + gap*3)/2, 100))
    screen.blit(score_surface, score_rect)

    # 2. 보드 배경 (테두리 역할)
    board_w = tileSize * 4 + gap * 5
    board_h = tileSize * 4 + gap * 5
    pygame.draw.rect(screen, BOARD_BG, (startX - gap, startY - gap, board_w, board_h), border_radius=6)

    # 3. 타일 그리기
    for i in range(4):
        for j in range(4):
            val = board[i][j]
            x = startX + j * (tileSize + gap)
            y = startY + i * (tileSize + gap)
            
            # 셀 배경색
            bg_color = colors.getTileBg(val)
            pygame.draw.rect(screen, bg_color, (x, y, tileSize, tileSize), border_radius=3)
            
            # 숫자 텍스트
            if val > 0:
                fg_color = colors.getTileFg(val)
                # 숫자가 크면 폰트 크기를 줄이는 효과
                font_size = 32 if val >= 1024 else 36 if val >= 128 else 42
                temp_font = pygame.font.SysFont("arial", font_size, bold=True)
                val_surface = temp_font.render(str(val), True, fg_color)
                val_rect = val_surface.get_rect(center=(x + tileSize/2, y + tileSize/2))
                screen.blit(val_surface, val_rect)

    # 4. 게임 오버 텍스트
    if gameOver:
        over_surface = title_font.render("GAME OVER", True, OVER_TEXT_COLOR)
        over_rect = over_surface.get_rect(center=(startX + (tileSize*4 + gap*3)/2, startY + board_h + 40))
        screen.blit(over_surface, over_rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((totalWidth, gameHeight))
    pygame.display.set_caption("2048: AI vs Player")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 28, bold=True)
    title_font = pygame.font.SysFont("arial", 48, bold=True)

    # --- 상태 초기화 (AI 및 플레이어 독립 보드) ---
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

    # AI 모델 로드 (파일이 없으면 랜덤으로 움직임)
    ai_model = None
    if os.path.exists("2048_ai_model.zip"):
        ai_model = DQN.load("2048_ai_model")
        print("AI 모델을 성공적으로 불러왔습니다!")
    else:
        print("AI 모델 파일을 찾을 수 없어 랜덤으로 동작합니다. 먼저 train.py를 실행하세요.")

    ai_move_timer = 0
    ai_move_delay = 200  # AI가 너무 빨리 움직이지 않도록 딜레이 설정 (밀리초 단위)

    mouse_start_x = 0
    mouse_start_y = 0
    is_dragging = False

    running = True
    while running:
        dt = clock.tick(60) # 60 FPS 유지

        # 1. 이벤트 처리 (Player 키보드 + 마우스 입력)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not player_gameOver:
                dir_str = None
                
                # [키보드 조작]
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: dir_str = "up"
                    elif event.key == pygame.K_DOWN: dir_str = "down"
                    elif event.key == pygame.K_LEFT: dir_str = "left"
                    elif event.key == pygame.K_RIGHT: dir_str = "right"
                
                # [마우스 드래그 시작]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_start_x, mouse_start_y = event.pos
                    is_dragging = True
                
                # [마우스 드래그 끝]
                elif event.type == pygame.MOUSEBUTTONUP:
                    if is_dragging:
                        is_dragging = False
                        mouse_end_x, mouse_end_y = event.pos
                        dx = mouse_end_x - mouse_start_x
                        dy = mouse_end_y - mouse_start_y
                        
                        # 최소 스와이프 길이(20) 체크
                        if abs(dx) > 20 or abs(dy) > 20:
                            if abs(dx) > abs(dy):
                                dir_str = "right" if dx > 0 else "left"
                            else:
                                dir_str = "down" if dy > 0 else "up"

                # 방향이 결정되었으면 로직 실행
                if dir_str:
                    sim = logic.simulateMove(player_board, dir_str)
                    if sim["changed"]:
                        player_board = sim["result"]
                        player_score += sim["scoreGain"]
                        logic.addRandomTile(player_board)
                        if logic.isGameOver(player_board):
                            player_gameOver = True

        # 2. AI 행동 처리 (시간 딜레이 기반)
        if not ai_gameOver:
            ai_move_timer += dt
            if ai_move_timer > ai_move_delay:
                ai_move_timer = 0
                
                if ai_model:
                    obs = get_obs(ai_board)
                    action, _ = ai_model.predict(obs, deterministic=True)
                else:
                    action = np.random.randint(4)
                
                directions = {0: "up", 1: "down", 2: "left", 3: "right"}
                ai_dir_str = directions[int(action)]
                
                sim = logic.simulateMove(ai_board, ai_dir_str)
                
                if sim["changed"]:
                    ai_board = sim["result"]
                    ai_score += sim["scoreGain"]
                    logic.addRandomTile(ai_board)
                    if logic.isGameOver(ai_board):
                        ai_gameOver = True
                else:
                    # 🚨 AI가 막힌 방향을 고집해서 멈추는 현상 해결 🚨
                    # 선택한 방향으로 보드 변화가 없으면 강제로 무작위 행동을 1회 실행하여 뚫어줍니다.
                    fallback_action = np.random.randint(4)
                    fallback_dir = directions[fallback_action]
                    sim_fallback = logic.simulateMove(ai_board, fallback_dir)
                    
                    if sim_fallback["changed"]:
                        ai_board = sim_fallback["result"]
                        ai_score += sim_fallback["scoreGain"]
                        logic.addRandomTile(ai_board)
                        if logic.isGameOver(ai_board):
                            ai_gameOver = True

        # 3. 화면 그리기
        screen.fill(BG_COLOR)
        pygame.draw.line(screen, (200, 200, 200), (gameWidth, 0), (gameWidth, gameHeight), 2)
        draw_board(screen, font, title_font, ai_board, startX_AI, ai_score, ai_gameOver, "AI (DQN)")
        draw_board(screen, font, title_font, player_board, startX_Player, player_score, player_gameOver, "Player")

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()