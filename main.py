import os
import time
import numpy as np
import pygame

import config
from agent import Agent2048
from env import Game2048Env
from ntuple_network import NTupleNetwork
from ui import Button, animate_transition, draw_game_over, render_scene


def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("2048 AI Battle")
    clock = pygame.time.Clock()

    env = Game2048Env()
    board = env.reset()

    net = NTupleNetwork()
    if os.path.exists(config.BEST_MODEL_PATH):
        net.load(config.BEST_MODEL_PATH)
    elif os.path.exists(config.MODEL_PATH):
        net.load(config.MODEL_PATH)
    else:
        print(f"[warning] model file not found: {config.BEST_MODEL_PATH}")
        print("[warning] untrained network will be used.")

    agent = Agent2048(env, net)

    restart_button = Button(config.BOARD_LEFT, 710, 140, 36, "Restart")
    ai_button = Button(config.BOARD_LEFT + 160, 710, 140, 36, "AI: ON")

    ai_enabled = True
    last_ai_time = time.time()

    dragging = False
    drag_start = None

    game_over = False
    running = True

    while running:
        best_tile = int(np.max(board))
        render_scene(screen, board, env.score, best_tile, ai_enabled, restart_button, ai_button)

        if game_over:
            draw_game_over(screen)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if restart_button.clicked(event.pos):
                    board = env.reset()
                    game_over = False
                    continue

                if ai_button.clicked(event.pos):
                    ai_enabled = not ai_enabled
                    ai_button.text = "AI: ON" if ai_enabled else "AI: OFF"
                    continue

                # board area drag start
                dragging = True
                drag_start = event.pos

            if event.type == pygame.MOUSEBUTTONUP and dragging and not game_over:
                end_pos = event.pos
                dx = end_pos[0] - drag_start[0]
                dy = end_pos[1] - drag_start[1]
                dragging = False

                action = None
                threshold = 20

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
                    old_board = board.copy()
                    board, game_over = env.step(action)
                    if not np.array_equal(old_board, board):
                        animate_transition(
                            screen,
                            old_board,
                            board,
                            env.score,
                            int(np.max(board)),
                            ai_enabled,
                            restart_button,
                            ai_button,
                        )

            if event.type == pygame.KEYDOWN and not game_over:
                action = None

                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                elif event.key == pygame.K_r:
                    board = env.reset()
                    game_over = False
                    continue
                elif event.key == pygame.K_a:
                    ai_enabled = not ai_enabled
                    ai_button.text = "AI: ON" if ai_enabled else "AI: OFF"
                    continue

                if action is not None:
                    old_board = board.copy()
                    board, game_over = env.step(action)
                    if not np.array_equal(old_board, board):
                        animate_transition(
                            screen,
                            old_board,
                            board,
                            env.score,
                            int(np.max(board)),
                            ai_enabled,
                            restart_button,
                            ai_button,
                        )

        if ai_enabled and not game_over:
            now = time.time()
            if now - last_ai_time >= config.AI_MOVE_INTERVAL:
                if config.USE_EXPECTIMAX_IN_GAME:
                    action = agent.expectimax_action(board, depth=config.EXPECTIMAX_DEPTH)
                else:
                    action, _, _ = agent.select_action(board)

                if action is not None:
                    old_board = board.copy()
                    board, game_over = env.step(action)
                    if not np.array_equal(old_board, board):
                        animate_transition(
                            screen,
                            old_board,
                            board,
                            env.score,
                            int(np.max(board)),
                            ai_enabled,
                            restart_button,
                            ai_button,
                        )

                last_ai_time = now

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()