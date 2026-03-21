# Reinforce_Learning_2048

2048 게임을 강화학습으로 학습하고, 학습된 모델로 AI vs USER 형태의 게임을 실행하는 프로젝트입니다.

## 주요 구성

- `train.py`  
  N-tuple Network 기반으로 2048 에이전트를 학습합니다.
- `main.py`  
  `pygame` 기반 GUI에서 AI와 유저가 게임을 플레이할 수 있습니다.
- `plot_training_graph.py`  
  학습 로그를 바탕으로 평균 점수, 최대 타일, 도달률 등의 그래프를 생성합니다.
- `config.py`  
  학습 파라미터, 모델 저장 경로, N-tuple 패턴, UI 설정 등을 관리합니다.
- `env.py`  
  2048 환경의 이동/병합/종료 판정을 담당합니다.
- `agent.py`  
  행동 선택 및 expectimax 기반 추론 로직을 담당합니다.
- `ntuple_network.py`  
  N-tuple value network 구현입니다.

## 실행 환경

- Python 3.10 이상 권장

## 설치

먼저 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

## 학습 방법

기존 학습 결과를 완전히 새로 시작하고 싶다면 `model/` 폴더 안의 체크포인트와 가중치 파일을 삭제한 뒤 실행하세요.

```bash
python train.py
```

학습 중에는 다음 파일들이 `model/` 폴더에 저장됩니다.

- `ntuple_checkpoint.pkl`
- `train_state.pkl`
- `ntuple_weights.pkl`
- `best_ntuple_weights.pkl`
- `train_log.csv`

## 학습 결과 그래프

학습 로그를 바탕으로 그래프를 생성합니다.

```bash
python plot_training_graph.py
```

생성되는 대표 결과물:

- `avg_score.png`
- `avg_max_tile.png`
- `reach_rate.png`
- `best_avg_score.png`

## 게임 실행

학습된 모델로 게임을 실행합니다.

```bash
python main.py
```

실행 후 모드를 선택할 수 있습니다.

- VS BATTLE
- AI SOLO
- USER SOLO

또한 난이도에 따라 다른 가중치 파일을 불러오도록 구성되어 있습니다.

## 프로젝트 특징

- 2048 환경 직접 구현
- N-tuple Network 기반 value estimation
- 체크포인트 저장 및 이어서 학습 지원
- 평균 점수 / 최대 타일 / 도달률 로그 저장
- `pygame` 기반 시각적 플레이 지원
- expectimax 기반 추론 지원

## 폴더/파일 구조 예시

```text
Reinforce_Learning_2048/
├── agent.py
├── config.py
├── env.py
├── main.py
├── ntuple_network.py
├── plot_training_graph.py
├── reward.py
├── symmetry.py
├── train.py
├── README.md
└── model/
```

## 참고

- 학습 로그와 체크포인트는 `config.py`의 경로 설정을 기준으로 저장됩니다.
- `main.py`는 로컬 GUI 실행용이므로, 웹 배포용 프로젝트가 아니라 데스크톱 실행 프로젝트입니다.
