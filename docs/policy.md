# Residual policy

`policy/`는 기준 궤적에 손목·관절 residual을 더하고 Isaac Sim에서 학습·평가합니다. `config/policy.json`의 RSI는 기본 OFF입니다. 중력 curriculum과 data augmentation은 ON이며 현재 기본 실행 길이는 4096 환경 / 2000 iterations입니다.

플로팅 환경의 지지면은 **30×30cm**, 윗면 **z=0**입니다. 기준 캔은 중앙 XY=(0,0)에서 시작합니다.
`config/policy.json`의 `surface`에서 조정하며, 두께 2cm의 유한한 box collider 하나로 구현합니다.
큰 책상·다리·로봇 받침대·하단 바닥은 생성하지 않습니다. 기존 XY 위치 증강 ±5cm와 reset noise는 유지합니다.
학습·평가·`scripts/physics.py floating`에 함께 적용되며 팔 환경은 별도 `config/workcell.json`을 사용합니다.
환경 변경은 다음 실행부터 적용되고 기존 큰 책상 환경의 checkpoint와 설정 hash가 다릅니다.

| 모듈 | 역할 |
|---|---|
| `env.py` | Isaac/PhysX 환경, Revo2·캔·접촉·reset |
| `task.py` | residual action, 보상, 종료 조건 |
| `observations.py` | actor/critic 관측과 history·delay |
| `trajectory.py` | 저장된 기준 궤적 검증과 시간·좌표 변환 |
| `reference.py` | GPU 기준 동작 보간과 증강 |
| `curriculum.py` | RSI sampling과 단계적 중력 |
| `randomization.py` | 물성 randomization과 외란 |
| `ppo.py` | 일반 RSL-RL PPO, checkpoint, export |
| `runner.py` | 학습·평가 실행 |
| `progress.py` | rollout 통계, 매 iteration 터미널 표와 ETA |
| `replay.py` | 평가 결과 비교 재생 |
| `math3d.py` | GPU 좌표·회전 계산 |

## 학습

저장소 루트에서 README의 Python 환경을 설정합니다. RL은 Isaac Sim 환경에서 `pip install -e '.[rl]'`에 해당하는 의존성이 필요합니다. 현재 머신은 Isaac Sim 6.0.1, RSL-RL 5.4.1을 사용합니다.

```bash
# 기본 4096 환경 / 2000 iterations, RSI OFF
"$PYTHON" scripts/policy.py --mode train --headless --skip-evaluation \
  --num-envs 4096 --iterations 2000 --output local/results/policy/train

# 환경 수·iteration 수를 지정하려면
"$PYTHON" scripts/policy.py --mode train --headless --skip-evaluation \
  --num-envs 256 --iterations 1000 --output local/results/policy/train_small

# 같은 설정에서 이어 학습
"$PYTHON" scripts/policy.py --mode train --headless --skip-evaluation \
  --checkpoint local/results/policy/train/policy.pt --output local/results/policy/train
```

RSI를 켜려면 `--enable-rsi`, augmentation을 끄려면 `--disable-augmentation`, 중력 curriculum을 끄려면 `--disable-gravity-curriculum`을 사용합니다. actor/critic 관측 크기는 67/88입니다. 자세한 수치는 `config/policy.json`이 기준입니다.

4096 환경은 source USD 환경 하나를 구성한 뒤 Isaac Cloner와 PhysX replication으로 생성합니다.
환경 간 충돌은 PhysX environment ID로 분리하며, reset과 startup 물성 randomization은 각 환경에 적용합니다.
`app ready`는 Isaac 앱이 준비됐다는 뜻이고 아직 학습 환경 초기화가 남아 있습니다.
`[startup]` 로그가 source 구성 → 복제 → articulation/object view → PhysX 초기화 → environment ready →
PPO buffer 생성 → 첫 rollout 수집 → 첫 최적화 순서로 진행 상태와 환경 생성 경과 시간을 표시합니다.
첫 rollout의 최적화가 끝난 뒤부터 매 iteration마다 정렬된 표가 출력됩니다.

- iteration/목표 iteration, 누적 transitions, 초당 transitions, 수집·PPO 업데이트 시간
- 현재 실행의 경과 시간, 평균 iteration 시간으로 계산한 남은 시간(ETA), 예상 총 시간
- value/surrogate loss, entropy, learning rate, action standard deviation
- step reward, 최근 종료된 최대 100개 episode의 return/길이, 실패·demo 종료·timeout 수
- 모든 추종·제약 오차와 높이의 평균/최대, 종료 조건 발생 비율, 보상 항목별 평균/최대
- 실제 중력과 단계, control step 수, RSI/augmentation 상태, RSI bin 확률, push 횟수

`steps/s`는 병렬 환경 전체의 transitions/초입니다. 물리 tick/초나 렌더링 FPS가 아닙니다.
보상 항목 표에는 weight와 control dt가 모두 반영되어 평균 항목의 합이 `Step reward`와 같습니다.
오차의 단위는 이름의 `_m`, `_rad`, `_m_s`, `_rad_s`를 따릅니다. boolean 지표의 평균은 발생 비율입니다.
episode가 아직 끝나지 않았으면 `pending`으로 표시합니다. 초기 random episode length로 잘린 episode도
실제로 관측한 return/길이로 집계하며, demo 끝까지 도달한 횟수를 파지 성공률로 해석하지 않습니다.
재개 실행의 목표는 기존 iteration + `--iterations`이며 ETA는 **이번 실행**의 완료 횟수로 계산합니다.
경과 시간은 초기 환경 생성/평가를 제외한 학습 루프부터 계산합니다. 이후 checkpoint 저장 비용은
다음 iteration의 누적 경과 시간과 ETA에 반영됩니다.

`training.jsonl`에는 모든 수치를 매 iteration 저장하고 TensorBoard/W&B에도 중첩 지표까지 기록합니다.
예전 한 줄 JSON 터미널 출력이 필요하면 `--console json`을 추가합니다. 기본값은 `--console pretty`입니다.

Headless 실행에서는 Isaac Lab의 학습용 앱처럼 Fabric의 transform/velocity/force-sensor/joint-state
출력을 끄고 CUDA일 때 GPU interop을 사용합니다. 관측은 PhysX tensor에서 직접 읽습니다.
`World.step(render=False)`와 별도로 적용하는 출력 설정입니다. rollout 수집과 PPO update 시간은
터미널에서 따로 확인할 수 있습니다. 성능 비교 및 물리 회귀 결과는 Git 제외 영역
`local/reports/iteration_speed/`에 보관합니다.
GUI 경로, 충돌·mimic·solver 반복 수·물리 dt·보상·중력 스케줄은 유지합니다.
실제 GPU dynamics/broadphase와 Fabric 출력 설정은 `run_metadata.json`의 `physics`에 기록합니다.

중력 단계는 **벡터 환경의 control step** 기준이며 병렬 환경 수 4096을 곱하지 않습니다.
현재 PPO rollout은 iteration당 24스텝으로, 1500 iterations = 36000 control steps에서
중력 범위가 `[9.81,9.81]`이 됩니다. 기존처럼 reset 때 월드 중력을 적용하므로
실제 반영은 해당 경계 이후 첫 reset입니다. 현재 최대 episode 길이 270스텝을 고려해도
늦어도 약 1512 iteration 안에 적용되고, 남은 약 488~500 iterations는 최대 중력으로 학습합니다.
4096 환경은 같은 물리 scene의 중력을 공유합니다. 손 중력은 계속 OFF이고 캔에는 이 중력이 적용됩니다.

| 완료한 iteration 수부터 | reset 시 중력 범위 (m/s²) |
|---|---|
| 0 | 0 |
| 400 | 0~1 |
| 500 | 0.5~2 |
| 600 | 1~3 |
| 700 | 2~4 |
| 800 | 3~5 |
| 900 | 4~6 |
| 1000 | 5~7 |
| 1100 | 6~8 |
| 1200 | 7~9 |
| 1300 | 8~9.81 |
| 1400 | 9~9.81 |
| 1500 | **9.81 고정** |

단계별 범위·reset 적용 방식은 유지하고 시작 시점만 단축한 사용자 요청 설정입니다.
원래 130000 control steps(약 5417 iterations)의 스케줄은 `local/archive/before_gravity_2000/`에 보관했습니다.

기준 입력은 `data/grounded/reference.npz`, 모델은 `assets/models/`, 로봇 USD는 `assets/USD/`, 캔은 `assets/can/can.usda`입니다. 현재 복합 원통의 두 collider와 새 50점 reference를 사용합니다. 형상/asset hash를 checkpoint contract에 포함하므로 이전 캔의 checkpoint를 그대로 사용할 수 없습니다. 바닥 좌표는 이미 데이터에 적용되어 있으므로 `world_frame=grounded_dataset`에서는 추가 배치를 하지 않습니다. 첫 캔 collision 밑면은 `z=0`이며 이전 임시 배치의 1 mm 간격도 제거했습니다. `scripts/ground.py`로 원본에서 재생성할 수 있습니다. 실제 학습 결과와 체크포인트는 기본 `local/results/policy/`에 저장되어 Git에서 제외됩니다. 현재 입력은 15~41번의 27자세, 5.2초입니다. RL에서만 이 27자세를 기존 9초 제어 horizon으로 재배정하며 마지막 목표는 41번 자세입니다. 제거된 42~50번은 RSI·증강·평가에도 사용하지 않습니다.

## 평가·export

```bash
"$PYTHON" scripts/policy.py --mode evaluate --headless --num-envs 2 \
  --checkpoint local/results/policy/train/policy.pt \
  --output local/results/policy/evaluation --export
```

기본 평가는 source PLAY 조건과 엄격한 고정 기준 추종을 함께 기록합니다. `--evaluation-protocol source` 또는 `strict`로 선택할 수 있습니다. JSON·NPZ·PNG와 비교 재생 HTML, 요청한 JIT/ONNX가 출력 폴더에 생성됩니다. 실제로 측정한 추종 오차와 실패를 확인해야 하며, 재생만으로 물리적 파지 성공을 판정하지 않습니다.

체크포인트는 기준 데이터·모델·설정·관측 규약을 검사합니다. 경로가 바뀐 이전 실행의 checkpoint도 설정 hash가 달라 거부될 수 있으므로 검사를 임의로 우회하지 않습니다. 구조 정리 중에는 학습을 실행하지 않습니다. 학습 없는 환경/learner 검증 도구는 `local/tools/check_policy.py`로 분리했습니다.

`scripts/physics.py floating`과 `policy.py`는 같은 REGRIND 방식 손목 자세 PD(`control.py`)를 사용합니다. 손 전체의 중력은 `hand_gravity: false`로 끄고, 힘은 root에, 토크는 링크 질량 비율로 나눕니다. 이전 미리보기 전용 중력 보상·목표 속도 feedforward는 제거했습니다. GUI도 매 물리 tick에 PD를 갱신하고 화면 갱신만 별도로 수행하므로 headless와 물리/제어 주기가 같습니다. RL의 headless 제어 수식·게인·보상과 RSI 기본 OFF, 월드/물체 중력 curriculum은 유지합니다. 동작 미리보기에서는 curriculum을 끄고 캔에 9.81 m/s²를 적용합니다. 손 중력 OFF가 캔 중력 OFF를 뜻하지 않습니다. 미리보기는 residual=0이며 학습 정책 성능이나 물리적 파지 성공을 검증한 결과가 아닙니다.

[방법론 출처와 독립 구현](PROVENANCE.md#residual-rl-출처와-독립-구현)을 참고하세요. 상세 과거 대조표·시험 결과는 Git 제외 영역 `local/reports/`에 보관합니다.
