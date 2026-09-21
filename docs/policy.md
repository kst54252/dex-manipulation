# Residual policy

RB3 + Revo2를 실제 연결한 환경에서 손목·손가락 residual을 학습하는 새 경로는
[`config/policy_arm.json`과 팔 학습 설명](arm_policy.md)을 참고한다. 기존 floating 정책을
팔에 추론하는 경로와 별개이며, 매 제어 tick의 IK 및 실제 팔 추종 결과가 PPO 학습에 들어간다.

`policy/`는 기준 궤적에 손목·관절 residual을 더하고 Isaac Sim에서 학습·평가합니다. `config/policy.json`의 RSI는 기본 ON이며 원본 floating Revo2의 균등 프레임 샘플링을 사용합니다. 중력 curriculum과 data augmentation은 ON이며 현재 기본 실행 길이는 4096 환경 / 1000 iterations입니다.

현재 recipe는 `regrind_revo2_floating_v8_contact_table_clearance_v1`입니다. 원본의 직접 residual target,
손/물체별 접촉 보정, 손가락 drive와 solver, RSI reset을 반영했습니다. 손목·관절 목표에
추가한 governor는 기본 OFF이며, 파지 전 residual을 임의로 0으로 만드는 단계 제한도 없습니다.
원본의 작은 손목 추종 보상은 그대로이므로 RL 손 자세가 레퍼런스와 일치한다는 보장은 없습니다.

2026-09-21부터 기본 학습에 자체 설계한 `table_safety` 보상과 명령 보호를 추가했습니다.
17개 손 collision mesh의 **모든 vertex를 포함하는 링크별 상자**로 상판 Z=0 여유를 계산합니다.
의미 키포인트 검사로 대체하지 않습니다. 실제 PhysX 링크 자세를 120Hz로 검사한 최소 여유와
보호 전 actor 목표를 FK한 여유에 각각 비용을 부과합니다. 명령 여유 4mm, 속도 선행 시간 20ms이며,
필요할 때 손목 Z만 올립니다. XY·회전·손가락 목표는 그대로이고 물체 pose를 강제로 옮기지 않습니다.
상자 및 무한한 수평 반공간 검사는 실제 유한 상판보다 보수적이며 동적 무충돌 보장은 아닙니다.

보상은 `m=0.004m`, 실제 최소 간격 `c`, 보호 전 목표 간격 `c_raw`에 대해
`-8 * clip((m-c)/m, 0, 2)^2 -4 * clip((m-c_raw)/m, 0, 3)^2`를 기존 합에 더한 뒤 dt를 곱합니다.
보호가 위험한 actor 명령을 숨겨 보상을 받게 하지 않도록 **보호 전 목표**에도 비용을 매깁니다.
새 항도 매 iteration 통계에 출력합니다. 원본 REGRIND 수식에서 가져온 추가 항은 아닙니다.

현재 보존된 두 정책의 가중치는 이전 보상으로 학습된 그대로입니다. 새 보상을 정책이 학습하려면
새 출력 폴더에 재학습해야 하며, 기존 checkpoint를 새 recipe로 그대로 resume하면 계약 검사에서 거절합니다.
`--mode play`는 기존 정책에도 명령 보호를 기본 적용하고 `--table-safety checkpoint`는 학습 당시 설정을 재현합니다.
`--mode evaluate`는 기본적으로 학습 당시 설정을 사용합니다. 보호 적용 평가에는 `--table-safety protect`를 지정합니다.

관측 phase·속도 reference와 물리·RSI 설정이 바뀌어 **이전 recipe 정책을 새 설정으로 그대로 이어 학습할 수 없습니다**.
새 학습은 아래 명령으로 별도 폴더에 저장합니다. 완료한 1000회 재학습 checkpoint는
`local/results/policy/recovery_20260920/train_contact_1000/policy.pt`이고 정상 중력 검증·남은 실패는
`local/results/policy/recovery_20260920/REPORT.md`에 기록했습니다.
원본에서 참고한 범위와 유지한 차이는 [출처 기록](PROVENANCE.md#원본-접근-동작과-rsi-정합-2026-09-20)에 있습니다.

현재 학습 입력은 `data/demo2/policy_reference.npz`입니다. `local/policy/reference.npz`는
기존 체크포인트 설정 해시를 보존하는 상대 링크입니다. 원본 성공 실행도 실제 물리
rollout에서 만든 접촉 reference를 사용했음을 확인했습니다. 여기서는 이 프로젝트가 직접 학습한
v7 rollout의 손-물체 상대 자세를 이용해 손목·손가락 reference만 준비합니다. **원래 물체 목표,
27개 프레임, 5.2초 입력 시간, 사람 손 점은 그대로 유지합니다.** 준비된 reference는 이제
`data/demo2/`에 포함됩니다. 체크포인트 전달 시 `config.resolved.json` 및 해당 reference도 함께
보관해야 합니다. 이전 준비용 rollout은 정책 정리 과정에서 삭제했고, 아래는 새 rollout을 사용할 때의 명령 형식입니다.

```bash
"$PYTHON" scripts/policy_reference.py \
  --rollout local/results/new_evaluation/trained.npz \
  --output local/results/new_policy_reference/reference.npz
```

이 명령은 기존 출력을 덮어쓰지 않습니다. 환경 index 0을 고정 사용하며, 원본 target 일치와
전체 collider·바닥 간격, 보간 구간 20개 지점, 관절 위치·입력 시간에 따른 속도 한계를 검사합니다.
실제 접촉의 작은 관통을 제거하는 SLSQP projection에서 solver 상태와 실제 제약 통과 여부를
별도로 기록합니다. 준비된 기하만으로 파지 성공을 주장하지 않으며 새 학습의 물리 평가가 필요합니다.

플로팅 환경의 지지면은 **30×30cm**, 윗면 **z=0**입니다. 기준 캔은 중앙 XY=(0,0)에서 시작합니다.
`config/policy.json`의 `surface`에서 조정하며, 두께 2cm의 유한한 box collider 하나로 구현합니다.
큰 책상·다리·로봇 받침대·하단 바닥은 생성하지 않습니다. 기존 XY 위치 증강 ±5cm와 reset noise는 유지합니다.
학습·평가·`scripts/physics.py floating`에 함께 적용되며 팔 환경은 별도 `config/workcell.json`을 사용합니다.
환경 변경은 다음 실행부터 적용되고 기존 큰 책상 환경의 checkpoint와 설정 hash가 다릅니다.

| 모듈 | 역할 |
|---|---|
| `env.py` | Isaac/PhysX 환경, Revo2·캔·접촉·reset |
| `task.py` | residual action, 보상, 종료 조건 |
| `table.py` | collision geometry 기반 여유, batched FK, 공통 상판 명령 보호 |
| `observations.py` | actor/critic 관측과 history·delay |
| `trajectory.py` | 저장된 기준 궤적 검증과 시간·좌표 변환 |
| `prepare.py` | 자체 물리 rollout의 접촉 자세를 별도 학습 입력으로 준비 |
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
# 기본 4096 환경 / 1000 iterations, RSI ON, GUI/학습 후 평가·재생 없음
"$PYTHON" scripts/policy.py --mode train --headless --skip-evaluation \
  --num-envs 4096 --iterations 1000 --output local/results/policy/table_safe_1000

# 환경 수·iteration 수를 지정하려면
"$PYTHON" scripts/policy.py --mode train --headless --skip-evaluation \
  --num-envs 256 --iterations 1000 --output local/results/policy/train_small

# 위의 새 학습이 끝난 뒤 같은 설정으로 추가 100회 이어 학습
"$PYTHON" scripts/policy.py --mode train --headless --skip-evaluation \
  --config local/results/policy/table_safe_1000/config.resolved.json --iterations 100 \
  --checkpoint local/results/policy/table_safe_1000/policy.pt \
  --output local/results/policy/table_safe_1000
```

RSI를 끄려면 `--disable-rsi`, augmentation을 끄려면 `--disable-augmentation`, 중력 curriculum을 끄려면 `--disable-gravity-curriculum`을 사용합니다. actor/critic 관측 크기는 **67/94**입니다. critic에 물체 선속도·각속도 6개를 포함합니다. 자세한 수치는 `config/policy.json`이 기준입니다.
`--iterations`는 실행 길이만 정하며 중력 단계 시점을 자동으로 줄이지 않습니다.
실행에 사용한 설정은 출력 폴더의 `config.resolved.json`에도 저장합니다.

## 보상·증강·관절 구동

보상은 각 항을 더한 뒤 control dt=1/30초를 곱합니다. 거리와 속도는 실제 시뮬레이션 상태에서
계산하며 캔 pose는 reset 이외에는 쓰지 않습니다. 아래는 원본 floating Revo2와 맞춘 수식입니다.

| 항 | weight × 지수항 |
|---|---|
| 물체 50점 평균 거리 `d` | `1.5 exp(-d/0.02)` |
| 물체 선속도 오차 `dv` | `exp(-||dv||²/1²)` |
| 물체 각속도 오차 `dw` | `exp(-||dw||²/3.14²)` |
| 손목 위치 오차 `p` | `0.05 exp(-p/0.02)` |
| 손목 회전 오차 `r` | `0.05 exp(-r/0.2)` |
| raw action 크기 | `0.5 exp(-mean(a²))` — 이전 weight 0 수정 |
| raw action 변화 | `exp(-mean((a-a_prev)²)/0.5²)` |
| raw action 범위 | `exp(-sum(max(abs(a)-1,0)))` |
| 시연 종료 전 실패 | `-10` |

종료 조건에는 자체 workspace·coupling 검사도 포함되므로 원본과 종료 사건까지 같다는 뜻은 아닙니다.
상판 여유까지 만족하는 완전히 정적인 이상 상태의 합은 6.1이고 step reward는 6.1/30입니다.

위치 증강은 에피소드마다 XY ±5cm를 하나 뽑아 손·물체 전체 궤적에 끝까지 유지합니다.
actor/critic의 위치 입력에서도 같은 이동량을 빼 기준 좌표로 관측합니다. yaw 증강은 기본 0입니다.
이전 방식처럼 접근 중 증강 offset을 줄여 캔 목표가 원래 위치로 미끄러지게 만들지 않습니다.
`mode: fade`는 이전 checkpoint 재현을 위해 남겨 두며 현재 기본값은 `rigid_sequence`입니다.

관절 gain은 원본과 같은 SI **3 N·m/rad / 0.1 N·m·s/rad**, force drive,
최대 effort **0.5 N·m**입니다. leader와 follower 모두 PD target을 받고 coupling도 유지합니다.
USD angular drive에 쓸 때만 degree 단위로 변환하며 PhysX에서 읽은 SI gain을 초기화 시 검증합니다.
`mimic_schema_policy: asset`은 원본 USD의 `NewtonMimicAPI`를 유지하고 추가 PhysX mimic을 중복 적용하지 않습니다.
Isaac Sim 6에서 실제 schema 등록과 coupling을 확인합니다. 별도 `single` 모드는 Newton을 제거하고
동일 관계를 명시적 PhysX mimic 하나로 표현하며, 32환경 비교에서 같은 동작을 확인한 호환 경로입니다.
실제 사용한 schema는 metadata에 기록하고 지원하지 않는 런타임에서는 조용히 제약을 생략하지 않습니다.

시뮬레이터의 관절 속도 상한은 원본과 같은 **100 rad/s**이며 실제 Revo2의 허용 속도가 아닙니다.
원본처럼 residual target은 joint position limits로 clip하고 추가 명령 slew 제한은 기본 OFF입니다.
`joint_target_velocity_limit: true`로 추가 제한을 켤 수 있지만 학습 설정이 달라집니다.
측정 속도·coupling·joint limit 위반은 계속 기록하며 엄격 평가에는 USD 모델의 속도 기준을 적용합니다.
팔/실물용 trajectory는 별도의 IK·관절 속도·충돌 검증을 통과해야 합니다.

손목 PD는 300/30, 회전 PD는 3/0.3으로 유지하고, 힘은 원본처럼 `right_hand_base_link`에,
토크는 링크 질량 비율에 따라 적용합니다. 손목 잔차는 각 축 ±33.33mm, 회전 잔차는 각 축
±0.10667rad입니다. 위치 벡터 최대치는 약 57.74mm로, 접근 중에도 허용됩니다.
손/물체 관통 보정 속도는 각각 **5/2 m/s**, 물체 contact offset은 2mm입니다.
solver는 scene 최대 position/velocity **64/4**, 손 **32/2**, 물체 **16/2**입니다.

원본 성공 실행과 맞추기 위해 floating 학습의 self-collision은 OFF입니다. 물체·손·책상 간 충돌은 유지합니다.
30cm 지지면, 복합 원통 캔, XY ±5cm 증강과 15~41번 입력 자세는 유지합니다. 캔 COM randomization은
XYZ ±2/2/1mm, default joint offset은 0입니다. 원본 asset은 수정하지 않고 실행 stage에만 적용합니다.

## 팔에 연결한 학습 정책 재생

`--robot arm`은 floating checkpoint의 actor를 실제 RB3+Revo2 상태에 매 제어 tick 적용합니다.
`physics.py arm`은 정책을 읽지 않는 reference 재생이므로 학습 정책 확인에는 아래 명령을 사용합니다.

```bash
PYTHONPATH=src:.deps:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/wanjunkim/IsaacLab/.venv/bin/python scripts/policy.py \
  --mode play --robot arm \
  --config local/results/policy/recovery_20260920/train_contact_1000/config.resolved.json \
  --checkpoint local/results/policy/recovery_20260920/train_contact_1000/policy.pt \
  --arm-config config/ik.json --evaluation-protocol strict \
  --output local/results/policy/play_arm_contact_1000
```

기존 책상·베이스 배치와 −90도 회전은 `ik.json`이 지정한 workcell/alignment에서 읽습니다.
현재 기본 조립은 `rb3_revo2_vertical.usda`입니다. 원본과 동일한 마운트 부품을 사용해
손의 +Z가 link6 +Z와 일치하도록 장착하며, `rb3_vertical.json`의 새 flange→wrist 변환으로 IK를 풉니다.
위 −90도는 책상 위 동작 전체의 배치 회전이며 손 장착을 꺾는 회전이 아닙니다.
실제 손·캔 pose와 속도, fingertip을 학습 source 좌표로 역변환하고 동일한 관측 history·actor를
사용합니다. 손목 PD 목표를 학습 때의 손목 응답 모델에 넣고, 응답 pose를 base 좌표로 변환한 뒤
USD mount를 반영하는 strict IK를 풉니다. 정책의 원래 PD 목표와 실제 손목 pose를 같다고 가정하지 않습니다.
손가락 6축 residual은 같은 모델의 이름/coupling으로 articulation에 적용합니다.
제어 30Hz·물리 120Hz이며 RSI·증강·관측 잡음은 꺼지고 첫 frame에서 시작합니다.

팔은 USD drive와 중력 ON, 손은 학습 설정의 중력 OFF, 캔은 중력 9.81의 자유 강체입니다.
floating 손목 force/torque는 팔에 직접 적용하지 않습니다. 손의 질량·관성·관절 PD·마찰과 캔·책상
물성은 checkpoint 첫 환경에서 복원합니다. 손 collider는 body 이름과 실제 shape 수를 검증해 매핑하고,
팔 자체의 물성·USD drive gain/effort는 유지합니다. 팔에는 위치와 함께 제한을 통과한 관절 목표의
차분 속도 `(q_target - q_previous) / control_dt`를 전달합니다. IK 실패 시 속도 목표는 0입니다.
큰 책상의 크기·배치는 유지하면서 고정 kinematic 지지면, contact/solver 설정과 마찰을 학습에 맞춥니다.
checkpoint의 학습 계약은 그대로 검사하고, 배치 변환이나 팔 설정으로 이를 덮어쓰지 않습니다.

`config/ik.json`의 `policy_control`은 팔 정책 실행에만 적용합니다. 기본 `wrist_response=training_pd`는
checkpoint 질량·관성과 원래 floating USD의 carrier frame/COM을 읽어 120Hz에서 강체 합성 응답을
계산합니다. carrier는 응답 계산에만 사용하며 실제 팔에 추가하지 않습니다. 측정 손 접촉력과 body COM
기준의 근사 접촉 모멘트를 저역 통과 필터(alpha 0.5)에 넣고 0.25배 반영합니다. 이는 자체 구현한
전이용 근사 모델이며 손가락 내부 운동량과 정확한 접촉점 모멘트까지 재현하지 않습니다.
`wrist_response=direct`는 원래 PD 목표를 바로 IK에 보내는 비교용 모드,
`contact_feedback_gain=0`은 접촉 피드백 없는 응답 모델입니다. 기본 정책·보상·학습 가중치는 바꾸지 않습니다.

IK는 이전 성공 명령에서 시작해 실제 control dt의 arm velocity/step/position limits와
singularity를 검사합니다. 실행 중에는 한 개의 local seed, 초기화에는 기존 multistart를 씁니다.
도달 불가·큰 점프·singular target을 임의로 이동시키지 않으며, 실패하면 직전 arm/finger 명령을
유지한 채 해당 episode를 실패 종료합니다. 콘솔의 `[arm IK failure]` 및 `episodes.jsonl`에 남습니다.
`first_episode.npz`에는 arm actual/target, finger, 정책 행동, 실제 손/캔 pose와 IK/관절/추종 오차가 저장됩니다.
`policy_wrist_*`는 **원래 정책 PD 목표 대비 실제 손목**, `arm_tracking_*`는 **응답 모델을 거쳐
IK에 전달한 목표 대비 실제 팔 FK**, `arm_fk_consistency_*`는 **측정 관절 FK 대비 PhysX 손목**의 오차입니다.
이 세 지표를 혼동하지 않아야 합니다. 종료는 누적 float 오차 대신 명령 개수로 판단해 157개 명령을 실행합니다.
손-물체/책상 충돌은 PhysX에서 켜며, 자기 충돌은 checkpoint 설정을 따릅니다(v8은 OFF).
충돌 없는 궤적이나 팔 파지 성공을 보증하지 않습니다.

이 경로는 직접 목표 방식 checkpoint를 대상으로 하며 `train/evaluate`, source 잡음 재생,
추가 floating governor는 지원하지 않습니다. GUI를 종료하거나 `--episodes 1`로 반복을 제한할 수 있습니다.
초기 구현 검증은 simulator 없는 좌표/IK/관절명 매핑 테스트와 이전 actor의 270frame 기하 검사로 수행했습니다.
기하 검사에서는 can pose와 관절 추종을 이상적으로 가정하므로 동적 파지 검증이 아닙니다.
초기 CPU 검사 뒤 보고된 articulation 초기화 오류는 Isaac 런타임에서 수정·검증했습니다.
Newton schema가 손에 추가하는 implicit articulation root를 감지하고, 고정 base와 손까지의
활성 joint 연결을 확인한 뒤 실행 stage에서 중복된 손 root만 제거합니다. 원본 USD는 수정하지 않습니다.
실제 headless 정책 재생 기록은 `local/results/policy/arm_runtime_fix_20260920/`에 있으며,
GUI 표시와 파지 성공은 별도 항목입니다. v8의 link-origin fingertip index 초기화도 팔에 적용했습니다.
수정 전 실패와 수정 후 재검증·오차 그래프·한계는 Git 제외
[`arm_tracking_20260920/REPORT.md`](../local/results/policy/arm_tracking_20260920/REPORT.md)에 기록합니다.
재생 완료나 물체 궤적 오차 통과와 엄격한 손가락 제약 통과는 별도입니다. 현재 종료 후 유지 및
손가락 속도/coupling 제약의 실패가 남아 있으며, `tracking_success=false`를 성공으로 바꾸지 않습니다.

## RSI

`uniform_frames`는 현재 30Hz reference의 **0~155번 제어 프레임**을 동일 확률로 직접 선택하고
terminal 156번은 제외합니다. 제거한 원본 데이터 프레임을 복원하는 기능이 아닙니다.
선택한 frame의 손목 SE(3), 손 6개 독립 관절과 follower, 물체 SE(3), 각각의 속도를 함께 복원합니다.
속도는 원본처럼 제어 프레임의 이웃 자세에서 중앙 차분하고 처음은 0, 끝은 후진 차분합니다.
phase는 frame/(N-1)로 정규화합니다. 원본처럼 관절에만 ±0.02rad reset noise를 주고
손목·물체 pose에는 별도 reset noise를 주지 않습니다. 물체가 공중에 있는 frame의 RSI는
그 frame의 손·물체 상태 전체를 초기화하는 학습 기법이며, 파지 성공으로 집계하지 않습니다.

`uniform`은 이전 bin 기반 균등 sampler, `adaptive`는 이전 실패 분포 sampler로 보존합니다.
기존 sampler 이름을 바꿔 checkpoint를 임의로 resume하지 않습니다.
평가·재생에서는 RSI와 reset perturbation을 끄고 첫 프레임부터 전체 궤적을 실행합니다.

시각화를 금지한 학습은 위처럼 `--headless --skip-evaluation`을 함께 사용합니다. headless에서는 UI와
viewport 갱신도 명시적으로 끕니다. `--skip-evaluation`은 시작 전 baseline과 학습 후 평가·비교 재생 출력을
모두 건너뛰고 checkpoint 저장 후 종료합니다. Vizkit을 불러오는 코드는 없습니다. 이 명령은 학습만 수행하고 종료합니다. 별도 평가가 필요하면 `--headless`를 명시하며
`physics.py`, GUI 또는 뷰어를 자동 실행하지 않습니다.
실제 headless/render/viewport/평가 생략 여부는 `run_metadata.json`의 `execution`에 기록합니다.

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
이 Fabric 성능 변경 자체는 충돌·mimic·solver 반복 수·물리 dt·보상·중력 스케줄을 바꾸지 않습니다.
실제 GPU dynamics/broadphase와 Fabric 출력 설정은 `run_metadata.json`의 `physics`에 기록합니다.

중력 단계는 **벡터 환경의 control step** 기준이며 병렬 환경 수 4096을 곱하지 않습니다.
현재 PPO rollout은 iteration당 24스텝으로, 750 iterations = 18000 control steps에서
중력 범위가 `[9.81,9.81]`이 됩니다. reset 때 월드 중력을 적용하므로
실제 반영은 해당 경계 이후 첫 reset입니다. 현재 timeout 300스텝을 고려해도
늦어도 약 763 iteration 안에 적용되고, 남은 약 237~250 iterations는 최대 중력으로 학습합니다.
4096 환경은 같은 물리 scene의 중력을 공유합니다. 손 중력은 계속 OFF이고 캔에는 이 중력이 적용됩니다.

| 완료한 iteration 수부터 | reset 시 중력 범위 (m/s²) |
|---|---|
| 0 | 0 |
| 200 | 0~1 |
| 250 | 0.5~2 |
| 300 | 1~3 |
| 350 | 2~4 |
| 400 | 3~5 |
| 450 | 4~6 |
| 500 | 5~7 |
| 550 | 6~8 |
| 600 | 7~9 |
| 650 | 8~9.81 |
| 700 | 9~9.81 |
| 750 | **9.81 고정** |

사용자가 지정한 1000회 실행 길이에 맞춘 일정은 유지합니다. 이전 10000회 스케줄의 시점을 1/10로 줄여
초기 20%는 중력 0, 마지막 약 25%는 최대 중력으로 학습하는 비율을 유지합니다.
손목 중력 보상 힘을 추가한 것은 아닙니다. RSI, 증강과 물성 randomization은 ON입니다.
외란 push의 기존 시작 시점 130000스텝은 이번 24000스텝 학습 범위 밖이므로 push는 발생하지 않습니다.
100회마다 checkpoint를 저장합니다. 이전 설정은
`local/results/policy/stable_1000_20260920/before/config/policy.json`에 보관했습니다.
중력·제어 설정이 checkpoint 계약에 포함되므로 기존 정책을 임의로 이어 붙이지 않습니다.
학습 이후 평가는 동일 설정 snapshot과 checkpoint를 사용해 headless로만 수행합니다.

원본 기하 입력은 `data/demo2/grounded/reference.npz`, 현재 RL 접촉 입력은 `local/policy/reference.npz`, 모델은 `assets/models/`, 로봇 USD는 `assets/USD/`, 캔은 `assets/can/can.usda`입니다. 현재 복합 원통의 두 collider와 새 50점 reference를 사용합니다. 형상/asset hash를 checkpoint contract에 포함하므로 이전 캔의 checkpoint를 그대로 사용할 수 없습니다. 바닥 좌표는 이미 데이터에 적용되어 있으므로 `world_frame=grounded_dataset`에서는 추가 배치를 하지 않습니다. 첫 캔 collision 밑면은 `z=0`이며 이전 임시 배치의 1 mm 간격도 제거했습니다. `scripts/ground.py`로 원본에서 재생성할 수 있습니다. 실제 학습 결과와 체크포인트는 기본 `local/results/policy/`에 저장되어 Git에서 제외됩니다. 현재 입력은 15~41번의 27자세, 5.2초입니다. v8 RL은 5.2초의 입력 timestamp를 보존하고 157개 control frame으로 보간하며 마지막 목표는 41번입니다. 10초 episode timeout으로 궤적을 늘리지 않습니다. 이는 재생 속도 설정이며 원본 촬영 FPS를 추정한 값이 아닙니다. 제거된 42~50번은 RSI·증강·평가에도 사용하지 않습니다.

## 학습 정책 GUI 재생

```bash
PYTHONPATH=src:.deps:. /home/wanjunkim/IsaacLab/.venv/bin/python scripts/policy.py \
  --mode play --config local/results/policy/recovery_20260920/train_contact_1000/config.resolved.json \
  --checkpoint local/results/policy/recovery_20260920/train_contact_1000/policy.pt \
  --evaluation-protocol strict --output local/results/policy/play_contact_1000
```

Isaac Sim Kit GUI에서 floating Revo2 한 대를 실제 checkpoint의 평균 행동으로 반복 재생합니다.
창을 닫으면 종료하며, `--episodes 3`을 추가하면 세 episode 뒤 종료합니다.
`--headless --episodes 1`로 같은 정책을 화면 없이 확인할 수도 있습니다.
이 저장소에 별도 Vizkit 의존성은 없으며 위 명령은 Isaac Sim의 Kit viewport를 사용합니다.

추가 학습 없이 RSI OFF, 첫 프레임 시작, 물체 중력 9.81 m/s², 손 중력 OFF로 실행합니다.
기본 strict 재생은 증강·관측 잡음을 끄고 checkpoint의 첫 환경 물성값을 복원합니다.
`--evaluation-protocol source`는 기존 source PLAY의 증강·잡음과 1 m 실패 기준을 적용합니다.
기준 궤적의 속도는 checkpoint 설정을 따릅니다. v6까지는 약 9초, v7 진단은 프레임당 한 제어 tick, v8은 입력의 5.2초입니다. 캔은 episode 시작에만 배치하고 이후에는 물리로 움직입니다.
실패·종료 후 처음부터 재시작하고, 실패를 성공으로 표시하지 않습니다.
출력 폴더의 `playback.json`에 재생 조건, `episodes.jsonl`에 종료 원인·실제 오차,
`first_episode.npz`에 첫 episode의 행동·관측 위치·참조·오차를 저장합니다.
학습 출력 폴더와 재생 출력 폴더는 분리해야 합니다.
위 명령은 완료한 v8 정책을 해당 snapshot으로 재생하는 예시입니다. 이전
`train_10000/policy.pt`를 확인하려면 해당 실행 당시의 설정을 `--config`로 지정해야 합니다.
이 머신의 수정 전 설정은 `local/results/policy/diagnosis_20260920/before/config/policy.json`에 보존했습니다.
체크포인트의 데이터·모델·학습 설정 계약 검사는 유지합니다. 아래 명시적인 재생 제어 변경은
학습 계약과 구분하여 `run_metadata.json`의 `execution.motion_control`에 기록합니다.

### 플로팅 동작 안정화

`play`와 `evaluate`의 기본은 모두 `--motion-control checkpoint`로 학습 당시 제어를 그대로 사용합니다.
`--motion-control stable`을 명시한 경우에만 현재 설정의 governor를 켜고, 학습 때와 다름을 기록합니다.
학습은 설정 파일의 `motion_control`을 사용하며 원본 방식의 새 기본값은 OFF입니다.

```bash
# 별도 제어 변경 실험: 기존 정책에 현재 governor를 명시적으로 적용 (파지 성공 미검증)
PYTHONPATH=src:.deps:. /home/wanjunkim/IsaacLab/.venv/bin/python scripts/policy.py \
  --mode play --motion-control stable \
  --config local/results/policy/diagnosis_20260920/corrected/config.resolved.json \
  --checkpoint local/results/policy/diagnosis_20260920/corrected/policy.pt \
  --evaluation-protocol strict --output local/results/policy/play_stable
```

`control.MotionController`는 Isaac API 없이 Torch로 작동합니다. 원본 reference를 변경하지 않고,
정책의 원시 손목/손가락 목표를 물리 스텝마다 연속적인 명령으로 변환합니다. 제한은 다음과 같습니다.

| 대상 | 속도 | 가속도 | 응답 시간 상수 |
|---|---:|---:|---:|
| 손목 이동, 벡터 크기 | 0.18m/s | 1.5m/s² | 0.04s |
| 손목 회전, 벡터 크기 | 1.2rad/s | 6rad/s² | 0.04s |
| 각 손 독립 관절 | USD와 coupling에서 유도 | 6rad/s² | 0.04s |

회전은 XYZW quaternion의 짧은 회전 방향을 사용합니다. 관절은 limits에 닿기 전 제동하며,
종속 관절의 속도 제한도 독립 관절 제한에 반영합니다. reset한 환경의 명령 이력만 초기화합니다.
손목 수치는 이번 시연용 설계값이며 RB3 제조사 Cartesian 속도 사양이 아닙니다.
출력 명령을 팔에 적용하려면 별도의 strict IK, arm joint velocity/step/collision 검사가 필요합니다.

이전 v5 checkpoint의 손/캔 관통 보정 속도는 0.1m/s였고, v6 기본값은 손 5m/s·캔 2m/s입니다.
손-물체·책상 충돌, 마찰, 캔 중력을 유지합니다. floating self-collision은 v7부터 원본처럼 OFF입니다. 접촉 수치 설정은 PhysX 생성 전에 적용하며,
실행 중 USD를 수정하여 articulation view가 무효화되는 경로는 허용하지 않습니다.
목표 명령 제한이 접촉 중 실제 관절 속도 제한이나 비관통 인증을 뜻하지 않습니다.

`first_episode.npz`와 평가 rollout에 `raw_target_*`, `applied_target_*`를 함께 저장합니다.
`physics_time_s`는 실제 시뮬레이션 경과 시간이며, 기존 `time_s`는 해당 스텝의 reference 명령 시간입니다.
기존 1000회 정책은 다른 제어 조건에서 학습됐으므로, 안정화 재생의 동작 개선을 파지 성능
개선으로 해석하면 안 됩니다. v5 1000회 학습과 진단 결과는
`local/results/policy/stable_1000_20260920/`에 보존합니다. 새 제어 설정이 이미 저장된 가중치를
고쳐 주지는 않으므로 기존 정책의 큰 residual이 자동으로 사라진다고 해석하지 않습니다.

## 평가·export

```bash
"$PYTHON" scripts/policy.py --mode evaluate --headless --num-envs 128 \
  --config local/results/policy/stable_1000_20260920/config.resolved.json \
  --checkpoint local/results/policy/stable_1000_20260920/policy.pt \
  --output local/results/policy/evaluation --export
```

기본 평가는 source PLAY 조건과 엄격한 고정 기준 추종을 함께 기록합니다. `--evaluation-protocol source` 또는 `strict`로 선택할 수 있습니다. JSON·NPZ·PNG와 비교 재생 HTML, 요청한 JIT/ONNX가 출력 폴더에 생성됩니다. 실제로 측정한 추종 오차와 실패를 확인해야 하며, 재생만으로 물리적 파지 성공을 판정하지 않습니다.

평가 JSON의 `demo_completed_count`는 시연 끝까지 도달한 수,
`object_tracking_only_count`는 물체 추종·최종 높이 기준까지 통과한 수,
`success_count`는 여기에 매 스텝의 관절 위치·속도·coupling 제약까지 통과한 수입니다.
`maximum_lift_m`와 실제 link pose NPZ도 함께 기록합니다. 학습 설정과 평가 설정이 다르면
해당 checkpoint의 `config.resolved.json`을 `--config`로 지정합니다.

체크포인트는 기준 데이터·모델·설정·관측 규약을 검사합니다. 경로가 바뀐 이전 실행의 checkpoint도 설정 hash가 달라 거부될 수 있으므로 검사를 임의로 우회하지 않습니다. 학습 없는 환경/learner 검증 도구는 `local/tools/check_policy.py`로 분리했습니다.

`scripts/physics.py floating`과 `policy.py`는 같은 REGRIND 방식 손목 자세 PD(`control.py`)를 사용합니다. 손 전체의 중력은 `hand_gravity: false`로 끄고, 힘은 `right_hand_base_link`에, 토크는 링크 질량 비율로 나눕니다. 이전 미리보기 전용 중력 보상·목표 속도 feedforward는 제거했습니다. GUI도 매 물리 tick에 PD를 갱신하고 화면 갱신만 별도로 수행하므로 headless와 물리/제어 주기가 같습니다. 평가·미리보기에서는 학습 설정의 RSI와 증강을 끕니다. 동작 미리보기에서는 curriculum을 끄고 캔에 9.81 m/s²를 적용합니다. 손 중력 OFF가 캔 중력 OFF를 뜻하지 않습니다. 미리보기는 residual=0이며 학습 정책 성능이나 물리적 파지 성공을 검증한 결과가 아닙니다.

[방법론 출처와 독립 구현](PROVENANCE.md#residual-rl-출처와-독립-구현)을 참고하세요. 상세 과거 대조표·시험 결과는 Git 제외 영역 `local/reports/`에 보관합니다.

## 원본 성공 실행과의 차이 재검증 (v7)

현재 코드뿐 아니라 원본 `floating_stable_ground_5000/params/env.yaml`, `agent.yaml`, TensorBoard 기록을 대조했습니다.
원본의 38-frame reference는 30Hz command로 진행하며, 10초는 timeout일 뿐 궤적 길이가 아닙니다.
기존 `retime_for_control_horizon(episode_length_s, dt)` 호출은 이를 잘못 혼동했습니다.
`reference_timing`을 명시하고 reference 프레임 수, phase 분모, RSI 분포, episode timeout을 분리했습니다.
구 checkpoint는 해당 실행의 `config.resolved.json`을 사용하면 기존 타이밍 계약을 유지합니다.

원본 `enableExternalForcesEveryIteration=true`에 비해 기존 World의 실제 값은 false였습니다.
이 설정과 `gpuMaxNumPartitions=8`을 명시하고 적용값을 metadata에 기록합니다.
hand mass는 원본의 `right_.*` body만 0.9~1.1배, gains는 0.8~1.2배, object mass는 0.85~1.15배입니다.
물체 마찰 0.5~1.2, 책상 마찰 0.6~1.2, 기본 마찰 0.8을 맞췄습니다.
critic 손끝 관측은 원본처럼 touch link 원점을 사용하며, 21 semantic keypoint 자체는 변경하지 않습니다.

RSI ON, 직접 residual, 손목 PD 300/30·3/0.3, 손가락 3/0.1·0.5Nm·100rad/s,
원본 reward 가중치/수식과 PPO 구성은 유지합니다. 별도 governor·접근 구간 residual gating은 사용하지 않습니다.
중력은 200 iteration부터 올라가고 750 경계에서 9.81m/s²에 도달하여 마지막 250회를 정상 중력으로 학습합니다.

평가의 `full_horizon`은 실패한 뒤의 프레임까지 분모에 포함합니다. 기존의 실패 시점까지만 계산한
오차는 비교용으로 유지하되 단독 성능 지표로 쓰지 않습니다. 기본 보고 기준은 episode별 평균 5mm,
모든 프레임의 50-point 평균 오차 최대 25mm, 중도 실패 없음입니다. 실제 충족 여부는 결과 JSON을 확인해야 합니다.
같은 로봇이어도 현재 입력은 다른 retargeting 자세와 더 작은 사용자 지정 캔입니다. 원본 궤적·캔·가중치를 복사하지 않습니다.

v7의 27-frame/30Hz 재생은 시간 문제를 확인한 대조 실험입니다. v8은
`reference_timing: input_timestamps`로 원래 5.2초 재생 시간을 보존하고 30Hz에서 보간합니다.
현재 입력은 157 control frame이며 timeout은 별도로 300 tick입니다. 실제 촬영 FPS를 추정한 값은 아닙니다.
원본의 30Hz 입력을 그대로 처리한 원칙을, 이 프로젝트의 명시된 입력 시계에 적용합니다.

원본의 자동 reset 순서도 반영합니다. RSI로 실제 상태 k를 복원한 뒤 command manager가 k+1로 진행하므로, 자동 reset 직후의 다음 목표는 k+1입니다. 외부에서 호출한 초기 reset은 k를 유지합니다. 32개 실제 PhysX 환경에서 위치 복원과 다음 명령 프레임을 따로 검증했습니다. 책상 torsional patch의 자체 고정 반경(20/5mm)도 원본 기본값(0/0)으로 되돌렸습니다.

## 정책 궤적을 저장한 뒤 팔 추종 검증

`tracking.py`는 실시간 정책과 팔 제어의 영향을 분리합니다. 먼저 strict floating
정책을 한 번 재생해 기록하고, 손목 PD 목표와 손가락 목표를 오프라인 IK로
변환합니다. 이후 팔은 저장된 관절 목표만 물리로 따라갑니다.

```bash
# POLICY_CONFIG/CHECKPOINT는 같은 학습 실행의 파일,
# ARM_CONFIG는 그 학습 reference를 가리키는 팔 배치 설정이다.
python scripts/policy.py --mode play --robot floating --headless --episodes 1 \
  --evaluation-protocol strict --config "$POLICY_CONFIG" --checkpoint "$CHECKPOINT" \
  --output local/results/frozen/capture
python scripts/tracking.py plan --capture local/results/frozen/capture \
  --arm-config "$ARM_CONFIG" --kind command --time-scale 4 \
  --output local/results/frozen/plan
python scripts/tracking.py replay --reference local/results/frozen/plan/trajectory.npz \
  --checkpoint "$CHECKPOINT" --headless --repeats 3 --hold 1 \
  --output local/results/frozen/replay
```

각 명령은 Isaac 환경 Python과 `PYTHONPATH=src:.deps:.`를 사용합니다.
`plan`은 Isaac 없이 실행 가능합니다. `--time-scale`은 1 이상의 정수로,
제어 주기를 유지하며 저장 궤적만 감속합니다. reset 상태→첫 명령 구간도
포함하므로 30Hz의 40개 명령은 1.333초(4배 감속은 5.333초)입니다.
원래 reference의 timestamp 범위 1.3초와 물리 실행 시간을 구분합니다.

`--kind command`는 actor의 PD 목표 자체입니다. `--kind measured`는 별도
비교용으로 기록된 실제 floating wrist/finger 상태를 선택하며, 실제 상태가
관절 한계를 넘었으면 조용히 clip하지 않고 계획을 거부합니다.
실패한 IK/속도 제한 궤적은 `report.json`에 남고 재생은 거부됩니다.
통과하면 timestamp와 관절 이름을 가진 12-DoF `targets.csv`도 생성합니다.

재생 중 정책 추론·온라인 IK·가상 손목 응답은 실행하지 않습니다. checkpoint는
동일한 물성을 복원하는 데만 사용합니다. 초기화 이후 관절/물체 pose를 직접
덮어쓰지 않으며, 물체를 놓쳐도 전체 구간과 종료 후 유지를 기록합니다.
`run_*.npz`는 각 물리 tick의 목표/실제 wrist pose와 arm/finger 관절, 오차,
특이점 및 제한 위반을 보존합니다. 제어 주기 끝 오차와 모든 물리 tick의
오차는 각각 집계합니다. 고정 궤적 추종은 폐루프 파지 성공과 별개입니다.
