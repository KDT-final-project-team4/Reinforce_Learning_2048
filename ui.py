import time
import pygame

import config


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

pygame.font.init()
TITLE_FONT = pygame.font.SysFont("arial", 42, bold=True)
LABEL_FONT = pygame.font.SysFont("arial", 20, bold=True)
VALUE_FONT = pygame.font.SysFont("arial", 26, bold=True)
TILE_FONT = pygame.font.SysFont("arial", 34, bold=True)
GAME_OVER_FONT = pygame.font.SysFont("arial", 52, bold=True)
SUB_FONT = pygame.font.SysFont("arial", 22, bold=True)


class Button:
    def __init__(self, x, y, w, h, text):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text

    def draw(self, screen):
        pygame.draw.rect(screen, config.BUTTON_COLOR, self.rect, border_radius=10)
        label = LABEL_FONT.render(self.text, True, config.BUTTON_TEXT)
        screen.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


def tile_rect(row, col):
    board_size = config.BOARD_PIXEL_SIZE
    pad = config.GRID_PADDING
    tile = (board_size - pad * 5) // 4

    x = config.BOARD_LEFT + pad + col * (tile + pad)
    y = config.BOARD_TOP + pad + row * (tile + pad)
    return pygame.Rect(x, y, tile, tile)


def draw_header(screen, score, best_tile, ai_enabled):
    title = TITLE_FONT.render("2048 AI Battle", True, config.TEXT_DARK)
    screen.blit(title, (config.BOARD_LEFT, 28))

    score_box = pygame.Rect(config.BOARD_LEFT + 270, 22, 120, 60)
    best_box = pygame.Rect(config.BOARD_LEFT + 405, 22, 120, 60)
    mode_box = pygame.Rect(config.BOARD_LEFT + 540, 22, 100, 60)

    for box in [score_box, best_box, mode_box]:
        pygame.draw.rect(screen, config.PANEL_COLOR, box, border_radius=10)

    score_label = LABEL_FONT.render("SCORE", True, config.TEXT_LIGHT)
    best_label = LABEL_FONT.render("BEST TILE", True, config.TEXT_LIGHT)
    mode_label = LABEL_FONT.render("AI", True, config.TEXT_LIGHT)

    screen.blit(score_label, score_label.get_rect(center=(score_box.centerx, score_box.y + 16)))
    screen.blit(best_label, best_label.get_rect(center=(best_box.centerx, best_box.y + 16)))
    screen.blit(mode_label, mode_label.get_rect(center=(mode_box.centerx, mode_box.y + 16)))

    score_val = VALUE_FONT.render(str(score), True, config.TEXT_LIGHT)
    best_val = VALUE_FONT.render(str(best_tile), True, config.TEXT_LIGHT)
    mode_val = VALUE_FONT.render("ON" if ai_enabled else "OFF", True, config.TEXT_LIGHT)

    screen.blit(score_val, score_val.get_rect(center=(score_box.centerx, score_box.y + 42)))
    screen.blit(best_val, best_val.get_rect(center=(best_box.centerx, best_box.y + 42)))
    screen.blit(mode_val, mode_val.get_rect(center=(mode_box.centerx, mode_box.y + 42)))


def draw_board_base(screen):
    rect = pygame.Rect(
        config.BOARD_LEFT,
        config.BOARD_TOP,
        config.BOARD_PIXEL_SIZE,
        config.BOARD_PIXEL_SIZE,
    )
    pygame.draw.rect(screen, config.PANEL_COLOR, rect, border_radius=12)

    for r in range(4):
        for c in range(4):
            pygame.draw.rect(screen, config.EMPTY_TILE_COLOR, tile_rect(r, c), border_radius=10)


def draw_tile(screen, row, col, value, scale=1.0):
    rect = tile_rect(row, col)

    if value == 0:
        return

    color = TILE_COLORS.get(value, (60, 58, 50))

    if scale != 1.0:
        cx, cy = rect.center
        nw = max(10, int(rect.width * scale))
        nh = max(10, int(rect.height * scale))
        rect = pygame.Rect(0, 0, nw, nh)
        rect.center = (cx, cy)

    pygame.draw.rect(screen, color, rect, border_radius=10)

    font = TILE_FONT
    if value >= 1024:
        font = pygame.font.SysFont("arial", 28, bold=True)
    if value >= 16384:
        font = pygame.font.SysFont("arial", 22, bold=True)

    text_color = config.TEXT_DARK if value <= 4 else config.TEXT_LIGHT
    text = font.render(str(value), True, text_color)
    screen.blit(text, text.get_rect(center=rect.center))


def draw_board(screen, board):
    draw_board_base(screen)
    for r in range(4):
        for c in range(4):
            draw_tile(screen, r, c, int(board[r][c]), scale=1.0)


def draw_game_over(screen):
    overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill(config.OVERLAY_COLOR)
    screen.blit(overlay, (0, 0))

    text = GAME_OVER_FONT.render("GAME OVER", True, (180, 40, 40))
    sub = SUB_FONT.render("Press Restart to play again", True, config.TEXT_DARK)

    screen.blit(text, text.get_rect(center=(config.WINDOW_WIDTH // 2, 330)))
    screen.blit(sub, sub.get_rect(center=(config.WINDOW_WIDTH // 2, 385)))


def render_scene(screen, board, score, best_tile, ai_enabled, restart_button, ai_button):
    screen.fill(config.BACKGROUND_COLOR)
    draw_header(screen, score, best_tile, ai_enabled)
    draw_board(screen, board)
    restart_button.draw(screen)
    ai_button.draw(screen)


def animate_transition(screen, old_board, new_board, score, best_tile, ai_enabled, restart_button, ai_button):
    start = time.time()
    duration = config.ANIMATION_MS / 1000.0

    changed = []
    for r in range(4):
        for c in range(4):
            if int(old_board[r][c]) != int(new_board[r][c]) and int(new_board[r][c]) != 0:
                changed.append((r, c))

    while True:
        elapsed = time.time() - start
        t = min(1.0, elapsed / duration)

        screen.fill(config.BACKGROUND_COLOR)
        draw_header(screen, score, best_tile, ai_enabled)
        draw_board_base(screen)

        for r in range(4):
            for c in range(4):
                v = int(new_board[r][c])
                if v == 0:
                    continue

                if (r, c) in changed:
                    # pop animation
                    if t < 0.5:
                        scale = 0.8 + 0.4 * (t / 0.5)
                    else:
                        scale = 1.2 - 0.2 * ((t - 0.5) / 0.5)
                else:
                    scale = 1.0

                draw_tile(screen, r, c, v, scale=scale)

        restart_button.draw(screen)
        ai_button.draw(screen)
        pygame.display.flip()

        if t >= 1.0:
            break