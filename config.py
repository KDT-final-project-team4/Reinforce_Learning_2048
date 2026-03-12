import os

BOARD_SIZE = 4

# ----------------------------
# Training
# ----------------------------
NUM_EPISODES = 50000
ALPHA = 0.0001

# feature별 방문횟수 기반 local alpha
USE_VISIT_COUNT_ALPHA = False
MIN_LOCAL_ALPHA = 0.00005

# 안정성 확인을 위해 optimistic TD는 끔
USE_OPTIMISTIC_TD = False
POSITIVE_TD_SCALE = 1.0
NEGATIVE_TD_SCALE = 1.0

# ----------------------------
# Reward
# 실제 2048 score와 정렬되도록 merge score만 사용
# ----------------------------
REWARD_MERGE_MULT = 1.0
REWARD_EMPTY_BONUS = 0.0
REWARD_LOSE_PENALTY = 0.0

# ----------------------------
# N-tuple patterns
# 8 tuples, length 6
# ----------------------------
TUPLES = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)],
    [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)],
    [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)],
    [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
    [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1)],
    [(0, 3), (1, 3), (2, 3), (1, 2), (2, 2), (3, 2)],
]

# ----------------------------
# Eval / in-game AI
# ----------------------------
USE_EXPECTIMAX_IN_GAME = True
EXPECTIMAX_DEPTH = 1
AI_MOVE_INTERVAL = 0.5

# ----------------------------
# Paths
# ----------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "ntuple_weights.pkl")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_ntuple_weights.pkl")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "ntuple_checkpoint.pkl")
TRAIN_STATE_PATH = os.path.join(MODEL_DIR, "train_state.pkl")

CHECKPOINT_EVERY = 100
EVAL_WINDOW = 200

# ----------------------------
# UI
# ----------------------------
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 760
BOARD_PIXEL_SIZE = 560
BOARD_TOP = 140
BOARD_LEFT = 80
GRID_PADDING = 12
ANIMATION_MS = 110

BACKGROUND_COLOR = (250, 248, 239)
PANEL_COLOR = (187, 173, 160)
EMPTY_TILE_COLOR = (205, 193, 180)
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)
BUTTON_COLOR = (143, 122, 102)
BUTTON_TEXT = (255, 255, 255)
OVERLAY_COLOR = (255, 255, 255, 190)