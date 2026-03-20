import os
import time
from typing import Tuple

import numpy as np
import streamlit as st

import config
from agent import Agent2048
from env import Game2048Env
from ntuple_network import NTupleNetwork


st.set_page_config(
    page_title="2048 RL Web",
    page_icon="🎮",
    layout="wide",
)


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

TEXT_DARK = "#776E65"
TEXT_LIGHT = "#F9F6F2"
BACKGROUND = "#FAF8EF"
PANEL = "#BBADA0"
EMPTY_TILE = "#CDC1B4"

WEIGHT_FILES = {
    "EASY": "models/best_ntuple_weights_10000.pkl",
    "MEDIUM": "models/best_ntuple_weights_30000.pkl",
    "HARD": "models/best_ntuple_weights_50000.pkl",
    "MASTER": "models/best_ntuple_weights_70000.pkl",
}

MODE_OPTIONS = {
    "VS": "VS BATTLE",
    "AI_SOLO": "AI SOLO",
    "USER_SOLO": "USER SOLO",
}

DIRECTIONS = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3,
}


st.markdown(
    """
    <style>
        .stApp {
            background: #faf8ef;
        }
        .board-wrap {
            background: #bbada0;
            padding: 14px;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.10);
        }
        .board-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(64px, 1fr));
            gap: 12px;
        }
        .tile {
            aspect-ratio: 1 / 1;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: clamp(20px, 2vw, 34px);
        }
        .score-card {
            background: #bbada0;
            color: #f9f6f2;
            border-radius: 14px;
            padding: 12px 16px;
            text-align: center;
            margin-bottom: 10px;
        }
        .score-title {
            font-size: 13px;
            opacity: 0.9;
        }
        .score-value {
            font-size: 28px;
            font-weight: 800;
            line-height: 1.1;
        }
        .small-note {
            color: #776e65;
            font-size: 14px;
        }
        .title-badge {
            display: inline-block;
            background: #8f7a66;
            color: #fff;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 14px;
            margin-bottom: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_network(difficulty: str) -> Tuple[NTupleNetwork, str]:
    net = NTupleNetwork()
    candidates = [
        WEIGHT_FILES.get(difficulty, ""),
        config.BEST_MODEL_PATH,
        config.MODEL_PATH,
    ]

    for path in candidates:
        if path and os.path.exists(path):
            net.load(path)
            return net, path

    return net, "UNTRAINED"


def init_state(mode: str, difficulty: str) -> None:
    st.session_state.mode = mode
    st.session_state.difficulty = difficulty
    st.session_state.ai_env = Game2048Env()
    st.session_state.user_env = Game2048Env()
    st.session_state.ai_board = st.session_state.ai_env.reset()
    st.session_state.user_board = st.session_state.user_env.reset()
    st.session_state.ai_done = False
    st.session_state.user_done = False
    st.session_state.status = "새 게임이 시작되었습니다."
    st.session_state.last_loaded_difficulty = difficulty


if "mode" not in st.session_state:
    init_state("VS", "MEDIUM")

if "last_loaded_difficulty" not in st.session_state:
    st.session_state.last_loaded_difficulty = st.session_state.difficulty


if st.session_state.last_loaded_difficulty != st.session_state.difficulty:
    st.cache_resource.clear()
    st.session_state.last_loaded_difficulty = st.session_state.difficulty


net, loaded_model_path = load_network(st.session_state.difficulty)
ai_agent = Agent2048(st.session_state.ai_env, net)


def tile_style(value: int) -> str:
    rgb = TILE_COLORS.get(int(value), (60, 58, 50))
    text_color = TEXT_DARK if value <= 4 else TEXT_LIGHT
    return f"background: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); color: {text_color};"


def render_board(board: np.ndarray, title: str, score: int, done: bool, key_prefix: str) -> None:
    max_tile = int(np.max(board))
    st.markdown(f'<div class="title-badge">{title}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="score-card"><div class="score-title">SCORE</div><div class="score-value">{score}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="score-card"><div class="score-title">MAX TILE</div><div class="score-value">{max_tile}</div></div>',
        unsafe_allow_html=True,
    )

    html = ['<div class="board-wrap"><div class="board-grid">']
    for r in range(4):
        for c in range(4):
            value = int(board[r][c])
            text = "" if value == 0 else str(value)
            html.append(f'<div class="tile" style="{tile_style(value)}">{text}</div>')
    html.append("</div></div>")

    st.markdown("".join(html), unsafe_allow_html=True)
    if done:
        st.error(f"{title} GAME OVER")
    else:
        st.caption(f"{key_prefix} 보드 진행 중")


def judge_vs() -> str:
    ai_score = st.session_state.ai_env.score
    user_score = st.session_state.user_env.score
    ai_tile = int(np.max(st.session_state.ai_board))
    user_tile = int(np.max(st.session_state.user_board))

    if user_score > ai_score:
        return "USER WIN"
    if ai_score > user_score:
        return "AI WIN"
    if user_tile > ai_tile:
        return "USER WIN"
    if ai_tile > user_tile:
        return "AI WIN"
    return "DRAW"


def perform_user_move(action: int) -> None:
    if st.session_state.user_done:
        st.session_state.status = "USER 보드는 이미 종료되었습니다."
        return

    sim = st.session_state.user_env.simulate_move(st.session_state.user_board, action)
    if not sim["valid"]:
        st.session_state.status = "이 방향으로는 이동할 수 없습니다."
        return

    st.session_state.user_board, st.session_state.user_done = st.session_state.user_env.step(action)
    st.session_state.status = "USER가 이동했습니다."

    if st.session_state.mode == "VS" and not st.session_state.ai_done:
        ai_action = ai_agent.expectimax_action(st.session_state.ai_board, depth=config.EXPECTIMAX_DEPTH)
        if ai_action is not None:
            st.session_state.ai_board, st.session_state.ai_done = st.session_state.ai_env.step(ai_action)
            st.session_state.status += " AI도 바로 응수했습니다."


def perform_ai_step() -> None:
    if st.session_state.ai_done:
        st.session_state.status = "AI 보드는 이미 종료되었습니다."
        return

    ai_action = ai_agent.expectimax_action(st.session_state.ai_board, depth=config.EXPECTIMAX_DEPTH)
    if ai_action is None:
        st.session_state.ai_done = True
        st.session_state.status = "AI가 더 이상 이동할 수 없습니다."
        return

    st.session_state.ai_board, st.session_state.ai_done = st.session_state.ai_env.step(ai_action)
    st.session_state.status = "AI가 한 수 진행했습니다."


def autoplay_ai(max_steps: int = 400) -> None:
    if st.session_state.ai_done:
        st.session_state.status = "AI 보드는 이미 종료되었습니다."
        return

    progress = st.progress(0, text="AI 자동 플레이 중...")
    for i in range(max_steps):
        ai_action = ai_agent.expectimax_action(st.session_state.ai_board, depth=config.EXPECTIMAX_DEPTH)
        if ai_action is None:
            st.session_state.ai_done = True
            break
        st.session_state.ai_board, st.session_state.ai_done = st.session_state.ai_env.step(ai_action)
        progress.progress(min((i + 1) / max_steps, 1.0), text=f"AI 자동 플레이 중... {i + 1} step")
        if st.session_state.ai_done:
            break
    progress.empty()
    st.session_state.status = "AI 자동 플레이가 끝났습니다."


with st.sidebar:
    st.title("🎮 2048 RL Web")
    selected_mode = st.selectbox("모드", list(MODE_OPTIONS.keys()), format_func=lambda x: MODE_OPTIONS[x], index=list(MODE_OPTIONS.keys()).index(st.session_state.mode))
    selected_difficulty = st.selectbox("난이도", ["EASY", "MEDIUM", "HARD", "MASTER"], index=["EASY", "MEDIUM", "HARD", "MASTER"].index(st.session_state.difficulty))

    if st.button("새 게임 시작", use_container_width=True):
        init_state(selected_mode, selected_difficulty)
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.write("**로드된 모델**")
    if loaded_model_path == "UNTRAINED":
        st.warning("학습된 가중치 파일을 찾지 못해 untrained model로 실행 중입니다.")
    else:
        st.success(loaded_model_path)

    st.write("**조작 안내**")
    st.caption("- USER SOLO / VS: 아래 방향 버튼으로 이동")
    st.caption("- AI SOLO: AI 한 수 또는 자동 플레이")
    st.caption("- 난이도를 바꾼 뒤 새 게임 시작을 누르면 해당 모델로 다시 시작")


st.title("2048 Reinforcement Learning Web Demo")
st.caption("기존 강화학습 로직(env, agent, n-tuple network)은 유지하고, UI만 Streamlit으로 웹화한 버전입니다.")

info1, info2, info3 = st.columns(3)
info1.metric("MODE", MODE_OPTIONS[st.session_state.mode])
info2.metric("DIFFICULTY", st.session_state.difficulty)
info3.metric("EXPECTIMAX DEPTH", config.EXPECTIMAX_DEPTH)

if st.session_state.mode != selected_mode or st.session_state.difficulty != selected_difficulty:
    st.info("사이드바에서 선택한 설정을 적용하려면 '새 게임 시작'을 눌러주세요.")

st.markdown(f"<div class='small-note'>상태: {st.session_state.status}</div>", unsafe_allow_html=True)


if st.session_state.mode == "VS":
    left, right = st.columns(2)
    with left:
        render_board(st.session_state.ai_board, "AI", st.session_state.ai_env.score, st.session_state.ai_done, "AI")
    with right:
        render_board(st.session_state.user_board, "USER", st.session_state.user_env.score, st.session_state.user_done, "USER")

    st.subheader("USER 조작")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("⬆️ UP", use_container_width=True):
        perform_user_move(DIRECTIONS["UP"])
        st.rerun()
    if c2.button("⬇️ DOWN", use_container_width=True):
        perform_user_move(DIRECTIONS["DOWN"])
        st.rerun()
    if c3.button("⬅️ LEFT", use_container_width=True):
        perform_user_move(DIRECTIONS["LEFT"])
        st.rerun()
    if c4.button("➡️ RIGHT", use_container_width=True):
        perform_user_move(DIRECTIONS["RIGHT"])
        st.rerun()

    if st.session_state.ai_done and st.session_state.user_done:
        st.success(f"최종 결과: {judge_vs()}")

elif st.session_state.mode == "AI_SOLO":
    render_board(st.session_state.ai_board, "AI SOLO", st.session_state.ai_env.score, st.session_state.ai_done, "AI")
    c1, c2 = st.columns(2)
    if c1.button("AI 한 수", use_container_width=True):
        perform_ai_step()
        st.rerun()
    if c2.button("AI 자동 플레이", use_container_width=True):
        autoplay_ai()
        st.rerun()

else:
    render_board(st.session_state.user_board, "USER SOLO", st.session_state.user_env.score, st.session_state.user_done, "USER")
    st.subheader("USER 조작")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("⬆️ UP", use_container_width=True):
        perform_user_move(DIRECTIONS["UP"])
        st.rerun()
    if c2.button("⬇️ DOWN", use_container_width=True):
        perform_user_move(DIRECTIONS["DOWN"])
        st.rerun()
    if c3.button("⬅️ LEFT", use_container_width=True):
        perform_user_move(DIRECTIONS["LEFT"])
        st.rerun()
    if c4.button("➡️ RIGHT", use_container_width=True):
        perform_user_move(DIRECTIONS["RIGHT"])
        st.rerun()

st.divider()
with st.expander("배포 팁", expanded=False):
    st.markdown(
        """
        - **Streamlit Community Cloud**: repo 연결 후 `app.py`를 엔트리포인트로 지정하면 바로 배포됩니다.
        - **Render**: `render.yaml`이 포함되어 있어 새 Web Service 생성 시 자동 설정이 가능합니다.
        - 학습 가중치가 repo 안에 없으면 웹은 동작하지만 AI는 untrained model로 실행됩니다.
        """
    )
