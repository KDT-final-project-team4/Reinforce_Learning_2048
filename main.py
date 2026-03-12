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

AI_IDLE_INTERVAL = 0.8           # 유저가 입력 안 해도 AI가 가끔 진행
AI_AFTER_USER_DEAD_INTERVAL = 0.5  # 유저 탈락 후 AI 자동 속도


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


def draw_center_info(ai_enabled, user_dead, ai_dead):
    cx = WIDTH // 2

    text1 = LABEL_FONT.render("R: Restart", True, TEXT_DARK)
    text2 = LABEL_FONT.render(f"AI AUTO: {'ON' if ai_enabled else 'OFF'} (A)", True, TEXT_DARK)
    text3 = LABEL_FONT.render("Arrow Keys / Mouse Swipe = USER move", True, TEXT_DARK)

    screen.blit(text1, text1.get_rect(center=(cx, 90)))
    screen.blit(text2, text2.get_rect(center=(cx, 120)))
    screen.blit(text3, text3.get_rect(center=(cx, 150 - 20)))

    status = []
    if user_dead:
        status.append("USER DEAD")
    if ai_dead:
        status.append("AI DEAD")

    if status:
        s = LABEL_FONT.render(" | ".join(status), True, (180, 40, 40))
        screen.blit(s, s.get_rect(center=(cx, 690)))


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
    restart = LABEL_FONT.render("Press R to restart", True, TEXT_DARK)

    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20)))
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 35)))
    screen.blit(restart, restart.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 80)))


def try_user_move(action, user_env, ai_env, ai_agent, ai_board, user_board, ai_dead, user_dead):
    moved_user = False

    old_user = user_board.copy()
    user_board, user_dead = user_env.step(action)
    if not np.array_equal(old_user, user_board):
        moved_user = True

    # 유저가 한 번 움직이면 AI도 한 번 같이 움직임
    if moved_user and not ai_dead:
        ai_action = ai_agent.expectimax_action(ai_board, depth=config.EXPECTIMAX_DEPTH) \
            if getattr(config, "USE_EXPECTIMAX_IN_GAME", True) \
            else ai_agent.select_action(ai_board)[0]

        if ai_action is not None:
            ai_board, ai_dead = ai_env.step(ai_action)

    return ai_board, user_board, ai_dead, user_dead


def main():
    ai_env = Game2048Env()
    user_env = Game2048Env()

    ai_board = ai_env.reset()
    user_board = user_env.reset()

    net = NTupleNetwork()
    if os.path.exists(config.BEST_MODEL_PATH):
        net.load(config.BEST_MODEL_PATH)
    elif os.path.exists(config.MODEL_PATH):
        net.load(config.MODEL_PATH)
    else:
        print("[warning] trained model not found. untrained model will be used.")

    ai_agent = Agent2048(ai_env, net)

    ai_enabled = True
    user_dead = False
    ai_dead = False
    dragging = False
    drag_start = None
    last_ai_time = time.time()

    running = True
    while running:
        screen.fill(BACKGROUND)

        draw_board(LEFT_BOARD_X, "AI", ai_board, ai_env.score, ai_dead)
        draw_board(RIGHT_BOARD_X, "USER", user_board, user_env.score, user_dead)
        draw_center_info(ai_enabled, user_dead, ai_dead)

        both_done = ai_dead and user_dead
        if both_done:
            draw_result(ai_env, user_env, ai_board, user_board)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    ai_env = Game2048Env()
                    user_env = Game2048Env()
                    ai_board = ai_env.reset()
                    user_board = user_env.reset()
                    ai_agent = Agent2048(ai_env, net)

                    user_dead = False
                    ai_dead = False
                    last_ai_time = time.time()
                    continue

                if event.key == pygame.K_a:
                    ai_enabled = not ai_enabled
                    continue

                if both_done:
                    continue

                if not user_dead:
                    action = None
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3

                    if action is not None:
                        ai_board, user_board, ai_dead, user_dead = try_user_move(
                            action, user_env, ai_env, ai_agent, ai_board, user_board, ai_dead, user_dead
                        )
                        last_ai_time = time.time()

            if event.type == pygame.MOUSEBUTTONDOWN:
                dragging = True
                drag_start = event.pos

            if event.type == pygame.MOUSEBUTTONUP and dragging:
                dragging = False

                if both_done or user_dead:
                    continue

                end_pos = event.pos
                dx = end_pos[0] - drag_start[0]
                dy = end_pos[1] - drag_start[1]
                threshold = 20

                action = None
                if abs(dx) > abs(dy):
                    if dx > threshold:
                        action = 3
                    elif dx < -threshold:
                        action = 2
                else:
                    if dy > threshold:
                        action = 1
                    elif dy < -threshold:
                        action = 0

                if action is not None:
                    ai_board, user_board, ai_dead, user_dead = try_user_move(
                        action, user_env, ai_env, ai_agent, ai_board, user_board, ai_dead, user_dead
                    )
                    last_ai_time = time.time()

        # -----------------------------
        # AI 자동 진행
        # -----------------------------
        if ai_enabled and not ai_dead and not both_done:
            now = time.time()

            # 유저가 살아있으면: 유저 입력이 없을 때만 천천히 따라감
            if not user_dead:
                if now - last_ai_time >= AI_IDLE_INTERVAL:
                    ai_action = ai_agent.expectimax_action(ai_board, depth=config.EXPECTIMAX_DEPTH) \
                        if getattr(config, "USE_EXPECTIMAX_IN_GAME", True) \
                        else ai_agent.select_action(ai_board)[0]

                    if ai_action is not None:
                        ai_board, ai_dead = ai_env.step(ai_action)
                    last_ai_time = now

            # 유저가 죽었으면: AI는 원래 자동 속도로 계속 진행
            else:
                if now - last_ai_time >= AI_AFTER_USER_DEAD_INTERVAL:
                    ai_action = ai_agent.expectimax_action(ai_board, depth=config.EXPECTIMAX_DEPTH) \
                        if getattr(config, "USE_EXPECTIMAX_IN_GAME", True) \
                        else ai_agent.select_action(ai_board)[0]

                    if ai_action is not None:
                        ai_board, ai_dead = ai_env.step(ai_action)
                    last_ai_time = now

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()