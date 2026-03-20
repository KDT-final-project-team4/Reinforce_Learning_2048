import os
import time
import numpy as np
import pygame

import config
from env import Game2048Env
from agent import Agent2048
from ntuple_network import NTupleNetwork


pygame.init()

WIDTH = 1280
HEIGHT = 760
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048 AI vs USER")

clock = pygame.time.Clock()

BACKGROUND = (250, 248, 239)
PANEL = (187, 173, 160)
EMPTY_TILE = (205, 193, 180)
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)
BUTTON = (143, 122, 102)
BUTTON_HOVER = (163, 142, 122)

TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (138, 43, 226),
    8192: (255, 20, 147),
}

WEIGHT_FILES = {
    "EASY": "best_ntuple_weights_10000.pkl",
    "MEDIUM": "best_ntuple_weights_30000.pkl",
    "HARD": "best_ntuple_weights_50000.pkl",
    "MASTER": "best_ntuple_weights_70000.pkl",
}

TITLE_FONT = pygame.font.SysFont("arial", 38, bold=True)
LABEL_FONT = pygame.font.SysFont("arial", 22, bold=True)
VALUE_FONT = pygame.font.SysFont("arial", 28, bold=True)
TILE_FONT = pygame.font.SysFont("arial", 30, bold=True)
RESULT_FONT = pygame.font.SysFont("arial", 50, bold=True)

BOARD_SIZE = 480
GRID_PADDING = 12
LEFT_BOARD_X = 70
RIGHT_BOARD_X = 730
BOARD_Y = 150
CENTER_X = WIDTH // 2 - BOARD_SIZE // 2

AI_IDLE_INTERVAL = 0.8           
AI_AFTER_USER_DEAD_INTERVAL = 0.5  


def tile_rect(board_x, row, col):
    tile = (BOARD_SIZE - GRID_PADDING * 5) // 4
    x = board_x + GRID_PADDING + col * (tile + GRID_PADDING)
    y = BOARD_Y + GRID_PADDING + row * (tile + GRID_PADDING)
    return pygame.Rect(x, y, tile, tile)


def draw_tile(board_x, row, col, value):
    rect = tile_rect(board_x, row, col)
    color = TILE_COLORS.get(int(value), (60, 58, 50))
    pygame.draw.rect(screen, color, rect, border_radius=10)

    if value != 0:
        font = TILE_FONT
        if value >= 1024:
            font = pygame.font.SysFont("arial", 24, bold=True)
        if value >= 10000:
             font = pygame.font.SysFont("arial", 20, bold=True)

        text_color = TEXT_DARK if value <= 4 else TEXT_LIGHT
        text = font.render(str(int(value)), True, text_color)
        screen.blit(text, text.get_rect(center=rect.center))


def draw_board(board_x, title, board, score, dead=False):
    title_text = TITLE_FONT.render(title, True, TEXT_DARK)
    screen.blit(title_text, (board_x, 35))

    score_box = pygame.Rect(board_x + 320, 35, 160, 60)
    pygame.draw.rect(screen, PANEL, score_box, border_radius=10)

    score_label = LABEL_FONT.render("SCORE", True, TEXT_LIGHT)
    score_value = VALUE_FONT.render(str(score), True, TEXT_LIGHT)

    screen.blit(score_label, score_label.get_rect(center=(score_box.centerx, score_box.y + 16)))
    screen.blit(score_value, score_value.get_rect(center=(score_box.centerx, score_box.y + 42)))

    outer = pygame.Rect(board_x, BOARD_Y, BOARD_SIZE, BOARD_SIZE)
    pygame.draw.rect(screen, PANEL, outer, border_radius=12)

    for r in range(4):
        for c in range(4):
            pygame.draw.rect(screen, EMPTY_TILE, tile_rect(board_x, r, c), border_radius=10)

    for r in range(4):
        for c in range(4):
            draw_tile(board_x, r, c, board[r][c])

    max_tile = int(np.max(board))
    info = LABEL_FONT.render(f"MAX TILE: {max_tile}", True, TEXT_DARK)
    screen.blit(info, (board_x, 645))

    if dead:
        dead_text = LABEL_FONT.render("GAME OVER", True, (180, 40, 40))
        screen.blit(dead_text, (board_x + 170, 645))


def draw_center_info(ai_enabled, user_dead, ai_dead, mode):
    cx = WIDTH // 2

    text1 = LABEL_FONT.render("R: Restart (Menu)", True, TEXT_DARK)
    text3 = LABEL_FONT.render("Arrow Keys / Mouse Swipe = USER move", True, TEXT_DARK)

    screen.blit(text1, text1.get_rect(center=(cx, 90)))
    
    if mode in ["VS", "AI_SOLO"]:
        text2 = LABEL_FONT.render(f"AI AUTO: {'ON' if ai_enabled else 'OFF'} (A)", True, TEXT_DARK)
        screen.blit(text2, text2.get_rect(center=(cx, 130)))
    
    if mode in ["VS", "USER_SOLO"]:
        screen.blit(text3, text3.get_rect(center=(cx, 720)))

    status = []
    if mode in ["VS", "USER_SOLO"] and user_dead:
        status.append("USER DEAD")
    if mode in ["VS", "AI_SOLO"] and ai_dead:
        status.append("AI DEAD")

    if status:
        s = LABEL_FONT.render(" | ".join(status), True, (180, 40, 40))
        screen.blit(s, s.get_rect(center=(cx, 680)))


def draw_result(ai_env, user_env, ai_board, user_board):
    ai_score = ai_env.score
    user_score = user_env.score
    ai_tile = int(np.max(ai_board))
    user_tile = int(np.max(user_board))

    if user_score > ai_score:
        result = "USER WIN"
    elif ai_score > user_score:
        result = "AI WIN"
    else:
        if user_tile > ai_tile:
            result = "USER WIN"
        elif ai_tile > user_tile:
            result = "AI WIN"
        else:
            result = "DRAW"

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((255, 255, 255, 180))
    screen.blit(overlay, (0, 0))

    text = RESULT_FONT.render(result, True, TEXT_DARK)
    sub = LABEL_FONT.render(
        f"AI score {ai_score} / tile {ai_tile}    |    USER score {user_score} / tile {user_tile}",
        True,
        TEXT_DARK,
    )
    restart = LABEL_FONT.render("Press R to return to Menu", True, TEXT_DARK)

    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 35)))
    screen.blit(restart, restart.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 80)))


def show_menu():
    modes = ["VS", "AI_SOLO", "USER_SOLO"]
    mode_labels = ["VS BATTLE", "AI SOLO", "USER SOLO"]
    difficulties = ["EASY", "MEDIUM", "HARD", "MASTER"]
    
    # 2048 타일 색상을 활용한 난이도별 색상 
    diff_colors = [
        (238, 228, 218),  # EASY (2 타일)
        (245, 149, 99),   # MEDIUM (16 타일)
        (246, 94, 59),    # HARD (64 타일)
        (138, 43, 226)    # MASTER (4096 타일 - 튀는 보라색)
    ]
    diff_text_colors = [TEXT_DARK, TEXT_LIGHT, TEXT_LIGHT, TEXT_LIGHT]
    
    mode_idx = 0
    diff_idx = 1
    
    # 패널 및 버튼 위치 설정
    panel_rect = pygame.Rect(WIDTH//2 - 380, HEIGHT//2 - 260, 760, 520)
    start_btn_rect = pygame.Rect(WIDTH//2 - 150, HEIGHT//2 + 150, 300, 65)
    
    mode_rects = [pygame.Rect(WIDTH//2 - 290 + (200 * i), HEIGHT//2 - 90, 180, 55) for i in range(3)]
    diff_rects = [pygame.Rect(WIDTH//2 - 330 + (170 * i), HEIGHT//2 + 40, 150, 55) for i in range(4)]

    running = True
    while running:
        screen.fill(BACKGROUND)
        mouse_pos = pygame.mouse.get_pos()
        
        # 메인 패널 그리기
        pygame.draw.rect(screen, PANEL, panel_rect, border_radius=20)
        
        # 타이틀
        title = RESULT_FONT.render("2048 AI GAME MODE", True, TEXT_LIGHT)
        screen.blit(title, title.get_rect(center=(WIDTH//2, HEIGHT//2 - 190)))
        
        # ---------------- MODE 선택 ----------------
        mode_title = LABEL_FONT.render("SELECT MODE", True, TEXT_LIGHT)
        screen.blit(mode_title, mode_title.get_rect(center=(WIDTH//2, HEIGHT//2 - 130)))
        
        for i, r in enumerate(mode_rects):
            is_selected = (i == mode_idx)
            is_hover = r.collidepoint(mouse_pos)
            
            color = TILE_COLORS[2048] if is_selected else (BUTTON_HOVER if is_hover else BUTTON)
            txt_color = TEXT_DARK if is_selected else TEXT_LIGHT
            
            pygame.draw.rect(screen, color, r, border_radius=12)
            label = VALUE_FONT.render(mode_labels[i], True, txt_color)
            screen.blit(label, label.get_rect(center=r.center))

        # -------------- DIFFICULTY 선택 --------------
        diff_title = LABEL_FONT.render("SELECT DIFFICULTY", True, TEXT_LIGHT)
        screen.blit(diff_title, diff_title.get_rect(center=(WIDTH//2, HEIGHT//2)))

        for i, r in enumerate(diff_rects):
            is_selected = (i == diff_idx)
            is_hover = r.collidepoint(mouse_pos)
            
            color = diff_colors[i] if is_selected else (BUTTON_HOVER if is_hover else BUTTON)
            txt_color = diff_text_colors[i] if is_selected else TEXT_LIGHT
            
            pygame.draw.rect(screen, color, r, border_radius=12)
            label = VALUE_FONT.render(difficulties[i], True, txt_color)
            screen.blit(label, label.get_rect(center=r.center))

        # --------------- START 버튼 ---------------
        start_hover = start_btn_rect.collidepoint(mouse_pos)
        start_color = (242, 177, 121) if start_hover else TILE_COLORS[1024]
        pygame.draw.rect(screen, start_color, start_btn_rect, border_radius=15)
        start_label = VALUE_FONT.render("START GAME", True, TEXT_DARK)
        screen.blit(start_label, start_label.get_rect(center=start_btn_rect.center))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(mode_rects):
                    if r.collidepoint(event.pos):
                        mode_idx = i
                for i, r in enumerate(diff_rects):
                    if r.collidepoint(event.pos):
                        diff_idx = i
                if start_btn_rect.collidepoint(event.pos):
                    running = False
                    
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                running = False
                
    return modes[mode_idx], difficulties[diff_idx]


def get_1d_mapping(line_vals):
    mapping = {}
    merged_indices = set()
    new_line = []
    target_idx = 0
    
    for orig_idx, val in line_vals:
        if len(new_line) > 0:
            prev_target, prev_val = new_line[-1]
            if prev_val == val and prev_target not in merged_indices:
                mapping[orig_idx] = prev_target
                new_line[-1] = (prev_target, prev_val * 2)
                merged_indices.add(prev_target)
                continue
        
        mapping[orig_idx] = target_idx
        new_line.append((target_idx, val))
        target_idx += 1
        
    return mapping

def get_movement_map(board, action):
    mapping = {}
    if action == 2: # LEFT
        for r in range(4):
            line = [(c, board[r][c]) for c in range(4) if board[r][c] != 0]
            map_1d = get_1d_mapping(line)
            for orig_c, new_c in map_1d.items():
                mapping[(r, orig_c)] = (r, new_c)
    elif action == 3: # RIGHT
        for r in range(4):
            line = [(c, board[r][c]) for c in range(3, -1, -1) if board[r][c] != 0]
            map_1d = get_1d_mapping(line)
            for orig_c, new_c in map_1d.items():
                mapping[(r, orig_c)] = (r, 3 - new_c)
    elif action == 0: # UP
        for c in range(4):
            line = [(r, board[r][c]) for r in range(4) if board[r][c] != 0]
            map_1d = get_1d_mapping(line)
            for orig_r, new_r in map_1d.items():
                mapping[(orig_r, c)] = (new_r, c)
    elif action == 1: # DOWN
        for c in range(4):
            line = [(r, board[r][c]) for r in range(3, -1, -1) if board[r][c] != 0]
            map_1d = get_1d_mapping(line)
            for orig_r, new_r in map_1d.items():
                mapping[(orig_r, c)] = (3 - new_r, c)
    return mapping


def animate_slide(redraw_background_fn, board_x, board, action, duration=0.1):
    start_time = time.time()
    tile_total_size = (BOARD_SIZE - GRID_PADDING * 5) // 4 + GRID_PADDING
    
    move_map = get_movement_map(board, action)

    while True:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / duration)
        
        redraw_background_fn()
        
        outer = pygame.Rect(board_x, BOARD_Y, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(screen, PANEL, outer, border_radius=12)
        for r in range(4):
            for c in range(4):
                pygame.draw.rect(screen, EMPTY_TILE, tile_rect(board_x, r, c), border_radius=10)

        screen.set_clip(outer)

        for r in range(4):
            for c in range(4):
                val = board[r][c]
                if val != 0:
                    target_r, target_c = move_map.get((r, c), (r, c))
                    dx = (target_c - c) * tile_total_size
                    dy = (target_r - r) * tile_total_size
                    
                    rect = tile_rect(board_x, r, c)
                    moving_rect = rect.copy()
                    moving_rect.x += int(dx * progress)
                    moving_rect.y += int(dy * progress)
                    
                    color = TILE_COLORS.get(int(val), (60, 58, 50))
                    pygame.draw.rect(screen, color, moving_rect, border_radius=10)
                    
                    font = TILE_FONT
                    if val >= 1024:
                        font = pygame.font.SysFont("arial", 24, bold=True)
                    if val >= 10000:
                        font = pygame.font.SysFont("arial", 20, bold=True)

                    text_color = TEXT_DARK if val <= 4 else TEXT_LIGHT
                    text = font.render(str(int(val)), True, text_color)
                    screen.blit(text, text.get_rect(center=moving_rect.center))

        screen.set_clip(None)
        pygame.display.flip()

        if progress >= 1.0:
            break


def main():
    while True:
        mode, difficulty = show_menu()

        ai_env = Game2048Env()
        user_env = Game2048Env()

        ai_board = ai_env.reset()
        user_board = user_env.reset()

        net = NTupleNetwork()
        weight_path = WEIGHT_FILES.get(difficulty, config.BEST_MODEL_PATH)
        
        if os.path.exists(weight_path):
            net.load(weight_path)
            print(f"Loaded weights: {weight_path}")
        elif os.path.exists(config.BEST_MODEL_PATH):
            net.load(config.BEST_MODEL_PATH)
            print(f"Weights not found. Loaded fallback: {config.BEST_MODEL_PATH}")
        elif os.path.exists(config.MODEL_PATH):
            net.load(config.MODEL_PATH)
            print(f"Weights not found. Loaded fallback: {config.MODEL_PATH}")
        else:
            print("[warning] no trained model files found. untrained model will be used.")

        ai_agent = Agent2048(ai_env, net)

        ai_enabled = True
        user_dead = False
        ai_dead = False
        dragging = False
        drag_start = None
        last_ai_time = time.time()

        def redraw_all():
            screen.fill(BACKGROUND)
            if mode == "VS":
                draw_board(LEFT_BOARD_X, "AI", ai_board, ai_env.score, ai_dead)
                draw_board(RIGHT_BOARD_X, "USER", user_board, user_env.score, user_dead)
            elif mode == "AI_SOLO":
                draw_board(CENTER_X, "AI SOLO", ai_board, ai_env.score, ai_dead)
            elif mode == "USER_SOLO":
                draw_board(CENTER_X, "USER SOLO", user_board, user_env.score, user_dead)
            
            draw_center_info(ai_enabled, user_dead, ai_dead, mode)

        running = True
        while running:
            redraw_all()
            
            both_done = False
            if mode == "VS":
                both_done = ai_dead and user_dead
                if both_done: draw_result(ai_env, user_env, ai_board, user_board)
            elif mode == "AI_SOLO":
                both_done = ai_dead
            elif mode == "USER_SOLO":
                both_done = user_dead

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        running = False 
                        continue

                    if event.key == pygame.K_a:
                        ai_enabled = not ai_enabled
                        continue

                if both_done:
                    continue

                if mode in ["VS", "USER_SOLO"] and not user_dead:
                    action = None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP: action = 0
                        elif event.key == pygame.K_DOWN: action = 1
                        elif event.key == pygame.K_LEFT: action = 2
                        elif event.key == pygame.K_RIGHT: action = 3

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        dragging = True
                        drag_start = event.pos

                    if event.type == pygame.MOUSEBUTTONUP and dragging:
                        dragging = False
                        dx = event.pos[0] - drag_start[0]
                        dy = event.pos[1] - drag_start[1]
                        threshold = 20
                        if abs(dx) > abs(dy):
                            if dx > threshold: action = 3
                            elif dx < -threshold: action = 2
                        else:
                            if dy > threshold: action = 1
                            elif dy < -threshold: action = 0

                    if action is not None:
                        sim = user_env.simulate_move(user_board, action)
                        if sim["valid"]:
                            board_x = RIGHT_BOARD_X if mode == "VS" else CENTER_X
                            animate_slide(redraw_all, board_x, user_board, action)
                            
                            user_board, user_dead = user_env.step(action)
                            redraw_all()
                            pygame.display.flip()

                            if mode == "VS" and not ai_dead:
                                ai_action = ai_agent.expectimax_action(ai_board, depth=config.EXPECTIMAX_DEPTH) \
                                    if getattr(config, "USE_EXPECTIMAX_IN_GAME", True) \
                                    else ai_agent.select_action(ai_board)[0]

                                if ai_action is not None:
                                    animate_slide(redraw_all, LEFT_BOARD_X, ai_board, ai_action)
                                    ai_board, ai_dead = ai_env.step(ai_action)
                                    redraw_all()
                                    pygame.display.flip()
                            
                            last_ai_time = time.time()

            if mode == "AI_SOLO" and ai_enabled and not ai_dead:
                now = time.time()
                if now - last_ai_time >= AI_IDLE_INTERVAL:
                    ai_action = ai_agent.expectimax_action(ai_board, depth=config.EXPECTIMAX_DEPTH) \
                        if getattr(config, "USE_EXPECTIMAX_IN_GAME", True) \
                        else ai_agent.select_action(ai_board)[0]

                    if ai_action is not None:
                        animate_slide(redraw_all, CENTER_X, ai_board, ai_action)
                        ai_board, ai_dead = ai_env.step(ai_action)
                    last_ai_time = now

            if mode == "VS" and user_dead and ai_enabled and not ai_dead:
                now = time.time()
                if now - last_ai_time >= AI_AFTER_USER_DEAD_INTERVAL:
                    ai_action = ai_agent.expectimax_action(ai_board, depth=config.EXPECTIMAX_DEPTH) \
                        if getattr(config, "USE_EXPECTIMAX_IN_GAME", True) \
                        else ai_agent.select_action(ai_board)[0]

                    if ai_action is not None:
                        animate_slide(redraw_all, LEFT_BOARD_X, ai_board, ai_action)
                        ai_board, ai_dead = ai_env.step(ai_action)
                    last_ai_time = now

            clock.tick(60)

if __name__ == "__main__":
    main()