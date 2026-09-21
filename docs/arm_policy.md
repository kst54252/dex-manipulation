# 팔을 포함한 residual RL

`config/policy_arm.json`은 RB3-730 + Revo2를 실제 조립한 고정 베이스 articulation에서
PPO를 학습한다. 새 학습 입력은 **1번 데모의 밑면을 수평으로 교정한 5.2초 버전**이다.
손목 residual은 policy-source 좌표계이며, 물체 배치는 `config/ik_demo1.json`의
베이스 중심에서 55cm, yaw +90도 변환을 사용한다. 손가락은 모델의 명시된 독립 관절 순서다.
과거 20iter 시험 정책은 수평 교정 전 입력과 `config/ik_original.json`을 유지한다.

| 항목 | 기존 floating 정책의 팔 실행 | 새 팔 환경 학습 |
|---|---|---|
| Actor 출력 | 손목 이동 3 + 회전 3 + 손가락 6 | 동일한 12개 residual |
| Actor / critic 관측 | 67 / 94 | 87 / 114 |
| 손목 목표 처리 | 가상 floating PD 응답 → CPU IK | 상판 보호 → batched strict IK |
| 실제 시뮬레이션 | RB3 + Revo2, 추론만 | RB3 + Revo2, 학습·평가·재생 모두 동일 adapter |
| 학습에 들어가는 팔 오차 | 없음 | IK 잔차, 실제 손목 추종, 실패·특이점·관절 한계 비용 |
| 팔 구동 | 관절 position / velocity target | 동일; floating root force·torque는 사용하지 않음 |

관측에 추가한 20개 값은 실제 arm q 6개, qdot 6개, 직전 IK position/rotation error 6개,
성공 여부 1개, 연속 실패 횟수 1개다. 관절·속도는 USD limits로 정규화한다.
IK 오차는 정책 좌표계로 변환하여 2cm / 0.2rad로 정규화한다. 기존 관측의 noise/delay와
별개로 추가 팔 상태는 현재 simulator 측정값이다.

30Hz 제어마다 USD 양쪽 joint frame, axis, flange-to-hand mount를 반영한 GPU FK/IK를 푼다.
직전 성공 관절로 시작하며 실제 dt의 관절 속도 제한과 최대 step을 적용한다. 위치와 회전을 함께
푸는 adaptive damped least squares이고, 후보를 FK로 다시 검사한다. 현재 설정의 허용치는
위치 0.01mm, 회전 0.0001rad, 최소 정규화 singular value 0.001이다. 전이 구간도 21개 지점에서 검사한다.
이는 전역 IK나 연속 시간 충돌 회피 증명이 아니다.

IK가 실패하면 arm/finger 모두 직전 목표를 유지한다. 실패 후보의 잔차는 그대로 보상·관측·로그에
남기고, 4번 연속 실패하면 종료한다. 실제 singularity 또는 관절 한계 위반도 종료 조건이다.
미분 가능한 IK는 필요하지 않다. PPO가 실제 팔 시뮬레이션에서 얻는 return으로 actor를 갱신한다.

기존 손·물체·상판 보상에 다음 비용을 더한 후 control dt를 곱한다.

- IK 위치/회전 비용: `-[clip(e_p/0.01,0,3)^2 + clip(e_r/0.1,0,3)^2]`.
- 실제 손목과 보호된 목표 간 추종 비용: `-[clip(e_p/0.02,0,3)^2 + clip(e_r/0.2,0,3)^2]`.
- IK 실패: 매 tick `-8`. 종료 비용은 기존 `-10`과 중복 계산하지 않는다.
- 정규화 Jacobian sigma가 near threshold 아래이면 제곱 비용, 관절 한계 warning 범위에서는 가중치 0.2의 제곱 비용.

RSI는 동일한 균등 프레임 방식이다. 해당 frame과 XY augmentation의 손목을 strict IK로 풀어
실제 팔을 초기화한다. 실패한 augmentation은 재표본화하고 마지막에는 명시적으로 nominal 배치를
시도한다. nominal도 실패하면 학습을 중단한다. rollout 중에는 물체 pose를 덮어쓰지 않는다.
손의 질량·마찰·gain, 캔 물성·COM과 상판 마찰은 기존 randomization을 유지한다. 팔의 질량과
USD gain/effort는 그대로이며 고정 베이스에는 floating root velocity push를 적용하지 않는다.

기본은 4096 환경 / 2000iter이다. 중력은 1500iter에서 9.81에 도달하고 마지막 500iter는 정상 중력이다.
world gravity는 팔과 캔에 적용되며 손 gravity는 기존대로 OFF다. 책상 크기·받침대는 workcell 설정을 따른다.
self-collision은 기존 설정 `false`를 유지한다. 손 상판 보호·접촉 물리만으로 arm-hand self-collision 검증을
했다고 해석하면 안 된다. 실제 로봇 구동은 이 작업에 포함하지 않는다.

## 실행

저장소 루트에서 다음과 같이 학습한다. UI와 학습 전후 자동 평가·시각화는 켜지지 않는다.

```bash
PYTHONPATH=src:.deps:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/wanjunkim/IsaacLab/.venv/bin/python scripts/policy.py \
  --mode train --robot arm --config config/policy_arm.json \
  --initialize-actor local/results/policy/original_slow_table_safe_2000_20260921_035416/policy.pt \
  --num-envs 4096 --iterations 2000 --headless --skip-evaluation \
  --output local/results/policy/original_arm_2000
```

`--initialize-actor`는 이전 floating actor의 평균 행동과 정규화 통계를 가져온다. 새 입력의 첫 layer
가중치는 0으로 시작하며 critic/optimizer는 새로 만든다. 학습률 0.0001, 초기 행동 표준편차 0.15는
팔 적응을 위한 자체 선택이다. reference·joint model·residual scale·상판 설정이 다른 checkpoint는
받지 않는다. `--initialize-actor`를 빼면 zero-residual actor부터 학습한다.

실행마다 생성되는 `config.resolved.json`, `run_metadata.json`, `training.jsonl`, `policy.pt`와
중간 checkpoint는 `local/`에 저장한다. 같은 팔 정책을 이어 학습할 때는 `--initialize-actor` 대신
`--checkpoint <기존 정책>` 및 그 옆의 `--config <config.resolved.json>`을 사용한다.
`--iterations`는 추가 iteration 수이며, 중력 schedule은 저장된 control step부터 이어간다.
물리 episode는 새로 reset하므로 중간 step까지 bitwise resume하는 기능은 아니다.

학습 후 사용자가 직접 화면을 켜려면:

```bash
./run.sh arm policy local/results/policy/original_arm_2000/policy.pt
```

새 팔 정책은 실제 팔 관측이 필요하므로 `floating` 실행을 거부한다. 학습 때와 같은 팔 배치·모델 및
IK 설정을 계약으로 검사한다. 재생은 학습 때의 상판 보호 설정을 기본 사용한다.
`first_episode.npz`에는 arm actual/target, wrist/finger 목표와 실측값, IK·추종·관절·특이점 오차를 남긴다.
초기 시험의 정상 실행을 파지 성능 개선으로 보고하지 않으며, full-horizon 평가와 실패 프레임은 별도다.

2026-09-21 구현 검증으로 4096 환경 / 20iter를 수행했다. 시험 checkpoint는 다음 명령으로 선택할 수 있다.
2000iter 본학습은 아직 실행하지 않았으며 이 checkpoint를 본학습 결과로 해석하면 안 된다.

```bash
./run.sh arm policy 1-arm-trial
```

검증 기록은 `local/reports/arm_training_20260921/REPORT.md`에 있다.
