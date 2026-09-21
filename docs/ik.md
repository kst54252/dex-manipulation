# RB3 arm IK와 12-DoF reference

`fk.py`의 `ArmModel`과 `ik.py`는 NumPy/SciPy만 사용합니다. 모델 추출에만 USD가 필요하며, Isaac 경계는 `sim.py`와 `scripts/replay.py`에 분리했습니다. RL·PD 튜닝·하드웨어 통신은 이 단계에 포함하지 않습니다.

## 모델과 좌표

`config/ik.json`이 실제 조립 USD와 base/flange/wrist prim을 지정합니다. 기본 조립은
`assets/USD/rb3_revo2_vertical.usda`, 추출 모델은 `assets/models/rb3_vertical.json`입니다.
원본 폴더와 동일한 `revo2_vertical_adapter` 부품으로 팔 끝과 손을 일직선 연결합니다.
90도 꺾인 이전 조립 `rb3_revo2.usd`와 추출 모델 `rb3.json`은 보존합니다.
USD 또는 장착을 바꾸면 **다시 추출하고 IK 궤적도 다시 생성**해야 합니다. 두 조립 모델을 섞어 쓰지 않습니다.

- Base: `/World/rb3_730es_u/Geometry/link0`.
- Flange: 마지막 arm rigid link인 `.../link6`. 제조사 flange-face나 TCP의 별도 원점을 추정하지 않습니다.
- Wrist: `/World/revo2_right/Geometry/world/right_hand_base_link`. semantic wrist keypoint와는 다른 프레임입니다.
- arm 관절 순서: `base, shoulder, elbow, wrist1, wrist2, wrist3`.
- 조립 USD의 실제 axis는 Z/Y/Y/Z/Y/Z입니다. 한계는 elbow ±150°, 나머지 ±360°, 속도 한계는 각 약 3.14 rad/s입니다. 이는 **USD에서 추출한 값**이며 실물 제조사 제한을 검증했다는 의미는 아닙니다.
- 기본 일자 조립의 link6→wrist translation은 약 `(0, 0, 0.141304971) m`, 회전은 identity입니다.
  손 base의 +Z와 link6 +Z가 일치합니다. 마운트 mesh는 link6 +Z의 100mm 부근에서 시작하여
  141.305mm까지 이어지고, visual과 collision이 모두 포함됩니다. 이 값은 USD에서 추출하며 코드에 하드코딩하지 않습니다.
- 이전 꺾인 조립은 translation 약 `(0.03, 0, 0.14) m`, Y축 회전 약 −90°였습니다.

AssemblerFixedJoint의 body 관계가 rigid link 대신 attachment Xform을 가리키므로, 양쪽 attachment→rigid-link 변환을 joint frame에 합성합니다. 이 변환과 authored zero pose를 대조합니다. 모든 계산은 m/rad, column-vector SE(3), quaternion은 XYZW입니다.

```
T_base_wrist_target = T_base_source @ T_source_wrist_target
T_base_flange_target = T_base_wrist_target @ inverse(T_flange_wrist)
```

원본 궤적은 카메라 좌표이며 현재 기본 입력은 `data/demo2/grounded/reference.npz`입니다. `scripts/ground.py`가 첫 캔의 밑면을 `z=0`, 초기 캔 로컬 +Z를 위쪽으로 만드는 고정 강체 변환을 손·캔 전체에 적용합니다. 모든 프레임의 관절·시간·상대 자세는 유지합니다. 현재 [복합 원통 캔](object.md)의 두 collider와 중심 오프셋을 반영하므로 첫 캔 pose 원점은 `z=0.0201195 m`입니다.

**카메라→RB3 베이스 실측 보정은 현재 없습니다.** 기본 `config/arm_ground_frame.json`의 `world_from_source`는 바닥 데이터 원점을 책상 좌표 `(0.55,0,0)` m에 놓고 Z축으로 −90° 회전하는 시뮬레이션 배치입니다. 캔 초기 pose 원점의 XY 거리는 베이스 중심에서 55cm이며 팔 정책 재생에도 같은 배치를 적용합니다. `config/workcell.json`의 로봇 장착면은 `z=-0.02m`이므로 `base_from_source = inverse(world_from_base) @ world_from_source`로 계산하며 베이스 기준 translation은 `(0.55,0,0.02)` m입니다. 손·캔의 world 목표 높이와 기존 방향은 유지합니다. 명시적 alignment JSON 없이는 solve가 중단됩니다.

실측 변환이 있다면 아래 형식의 JSON을 전달합니다. 실제 행렬로 채워야 합니다.

```text
{"base_from_source": <4×4 SE(3) 행렬>, "status": "measured_camera_to_base"}
```

위 카메라→베이스 행렬은 원본 카메라 reference를 입력할 때 사용합니다. 바닥 reference를 사용할 때는 `T_base_ground = T_base_camera @ T_camera_ground`로 합성해야 하며, `T_camera_ground`는 `data/demo2/grounded/frame.json`의 `source_from_world`입니다. 좌표계 변경만으로 하드웨어 보정이 검증되지는 않습니다.

바닥 데이터의 `place`는 설정된 배치와 world/base 변환을 복사합니다. 책상 world 좌표에서 바닥 데이터를 공중으로 옮기는 시뮬레이션 alignment는 CLI에서 거부합니다. 베이스 좌표의 +2cm는 실제 장착 높이에 대한 보정입니다. 원본 카메라 좌표를 명시적으로 입력할 때만 이전의 첫 손목→`FK(seed)` 배치를 사용합니다. 수치 seed는 실제 로봇의 현재 자세가 아닙니다.

## 재현 명령

저장소 루트에서 README의 `PYTHON`/`PYTHONPATH` 설정을 사용합니다.

```bash
# USD → standalone arm model
"$PYTHON" scripts/ik.py extract

# 원본 pose/reference → 바닥 좌표 데이터 (원본 유지)
"$PYTHON" scripts/ground.py

# 일부 프레임 검증
"$PYTHON" scripts/ik.py solve \
  --max-frames 3 --output local/results/ik/smoke

# 전체 궤적
"$PYTHON" scripts/ik.py solve \
  --output local/results/ik/full

# Isaac 창에서 관절 상태 기반 재생
"$PYTHON" scripts/replay.py --reference local/results/ik/full/trajectory.npz \
  --realtime --loops 3

# 원본 프레임 PNG도 저장하려면 --capture (visible viewport)
"$PYTHON" scripts/replay.py --capture --rate 30 --realtime \
  --output local/results/ik/replay_visible

# 화면 없이 실제 PhysX link pose와 FK 비교
"$PYTHON" scripts/replay.py --reference local/results/ik/full/trajectory.npz --headless
```

현재 배치는 `config/arm_ground_frame.json`, 초기 branch seed는 `config/ik.json`에 저장합니다. 이전 `local/results/ik/workspace_placement/`의 공중 배치는 현재 바닥 데이터용이 아닙니다. `--seed`는 USD 순서의 rad 값 6개이며 초기 branch 선택 기준입니다.

## Solver와 성공 기준

- 위치와 회전을 함께 최소화하는 bounded damped least squares를 사용합니다. 회전 오차는 `log(R_target R_FKᵀ)`의 rotation vector입니다.
- analytic spatial Jacobian을 수치 미분으로 검증했습니다. 특이점 SVD에는 이동 행을 0.5 m로 나눈 무차원 Jacobian을 사용합니다. 단위가 다른 위치/회전 행의 condition number를 그대로 섞지 않습니다.
- `σ_min < .03`일 때 damping을 증가시키며, `.001` 미만 또는 condition number > 10000인 해는 pose가 정확해도 실패입니다. 근접 경고는 `.03` 또는 condition > 200입니다.
- joint-limit 근처에는 정규화 가중치를 높이고 작은 중심/이전 자세 bias를 둡니다. 6축 팔의 위치+회전은 보통 여유 자유도가 없으므로 모든 자세에서 limit 회피가 가능하다고 가정하지 않습니다. 모든 후보는 엄격한 pose 허용오차를 통과해야 합니다.
- 첫 프레임은 결정적인 multi-start, 이후는 이전 성공 자세에서 warm start합니다. 허용 범위 안의 여러 후보 중 **실제 unwrapped 관절 차이**가 가장 작은 해를 선택합니다. 제한된 후보 탐색이며 전역 IK 완전 탐색은 아닙니다.
- 인접 해의 각 축 이동은 `min(20°, USD velocity × 실제 dt)`로 제한합니다. ±π 경계를 넘어간다는 이유로 ±2π 점프를 허용하지 않습니다.
- 기본 `trajectory_substeps=2`는 원본 0.2초 구간마다 중간 자세 하나를 추가합니다. 현재 15~41번의 27자세/시각을 그대로 유지하고 53개 IK를 0.1초 간격으로 풉니다. 위치·finger는 선형, wrist/object 회전은 SLERP이며 원본 knot 행렬과 관절값은 그대로 복사합니다. 보간은 관절 이동 제한을 완화하지 않고 촘촘히 검사하기 위한 설정입니다. 실행 시간은 현재 선택 구간의 5.2초입니다. `--max-frames`는 원본 프레임 수, `--trajectory-substeps`는 구간당 계산 횟수입니다.
- `FK(q)`로 다시 계산한 wrist 위치 오차 ≤ 1e-5 m, 회전 오차 ≤ 1e-4 rad, joint/continuity/singularity 조건을 모두 만족해야 success입니다.
- joint limit 경고는 5° 이내, hard margin은 1e-4 rad입니다. 원본 finger 위치·속도 제한도 검증합니다.
- 인접 관절 궤적을 20등분하여 특이점 지표를 확인합니다. 이는 유한 표본 검사이며 연속 시간의 수학적 비특이성 증명은 아닙니다. Cartesian 중간점은 joint interpolation의 FK이며, 별도 Cartesian 직선 추종을 보장하지 않습니다.
- 실패 후보도 오차와 사유를 저장합니다. 실패를 warm start로 쓰지 않으며 실패 이후 연결도 valid로 꾸미지 않습니다. 실패가 하나라도 있으면 전체 trajectory의 명령 소비를 거부합니다.

## 출력과 제어 경계

`trajectory.npz`와 `report.json`은 `local/results/ik/`에 저장합니다. Git에서 제외합니다.

`solve` 성공 시 `reference_12dof.csv`와 `reference_120hz.csv`도 함께 갱신합니다. metadata의 `scene_placement`에 world/base 변환과 workcell fingerprint를 저장하며, 재생 시 현재 환경과 다르면 재계산을 요구합니다.

| NPZ 필드 | 의미 |
|---|---|
| `timestamps_s`, `frame_ids` | 원본 timestamp·frame ID 유지; 삽입한 중간 표본의 frame ID는 −1 |
| `source_frame_mask`, `source_interval_frame_ids` | 원본/보간 표본 구분과 각 표본이 속한 원본 프레임 구간 |
| `q_arm`, `q_finger` | arm 6축, 독립 finger 6축, rad |
| `joint_position_rad`, `joint_names` | 이름 순서가 명시된 N×12 reference |
| `wrist_target_source`, `wrist_target_pose`, `flange_target_pose` | 원본/베이스 wrist 목표와 변환된 flange 목표 |
| `fk_pose`, `position_error_m`, `orientation_error_rad` | 재계산된 실제 기하 오차 |
| `sigma_min`, `condition`, `near_singular` | 특이점 지표 |
| `joint_limit_distance_rad`, `near_joint_limit` | 한계까지 거리·경고 |
| `joint_step_rad`, `arm_velocity_rad_s` | 실제 인접 후보의 이동·속도 |
| `success`, `transition_valid`, `failure_reason` | 성공·연결 여부와 실패 사유 |
| `damping_max`, `solver_iterations`, `candidate_count` | optimizer 진단 |
| `metadata_json` | 단위·좌표·alignment 상태·모델/입력 hash·제약 설정 |

`reference.py`의 `JointReference`는 순수 joint target만 반환합니다.

```python
from dex_manipulation.reference import JointReference
reference = JointReference("local/results/ik/full/trajectory.npz")
for target in reference.iter_targets(rate_hz=120):
    # target.timestamp_s, target.joint_names, target.position_rad
    # 추후 simulator/제조사/ROS adapter가 같은 named target을 소비
    pass
reference.export_csv("local/results/ik/full/reference_120hz.csv", rate_hz=120)
```

`JointTarget.ordered(names)`로 명시적인 순서 매핑을 합니다. `IsaacJointAdapter.set_target`은 articulation position target, `set_state`는 FK 검증용 상태 재생입니다. Revo2 종속 관절은 실제 coupling으로 6→11개를 확장합니다.

실물 연결 시 동일 reference 형식을 사용하되 **베이스/도구 보정, 제조사 joint zero·방향·이름·단위 매핑, 실물 제한, 시작 자세로의 진입 궤적**을 별도로 확인해야 합니다. 파일의 `hardware_calibrated`와 `initial_robot_state_verified`는 현재 false입니다. 현 구현은 하드웨어 패킷을 보내지 않습니다. 관절 선형 보간은 속도를 제한하지만 acceleration/jerk shaping은 아직 포함하지 않습니다.

## 검증 범위

개발 테스트는 `local/tests/test_ik.py`, 결과와 로그는 `local/reports/ik/`에 있습니다. Isaac 기본 재생은 중력·충돌·drive를 끄고 joint state를 적용하여 PhysX의 link pose를 FK와 비교합니다. **실제 PD 추종·파지 검증이 아닙니다.** `--mode targets`는 아래 물리 미리보기에서 사용합니다. 기존 USD drive로 수행하며 추종 성공이나 파지 성공을 보장하지 않습니다.

팔 자기 충돌·팔/손과 환경의 충돌, 실물 동역학·제어 성능은 검증하지 않았습니다. floating hand의 기존 비관통 검사만으로 결합 팔/손의 무충돌을 주장하지 않습니다. 방법의 출처·설계 선택은 [PROVENANCE.md](PROVENANCE.md)에 있습니다.


## 물리 동작 미리보기

```bash
"$PYTHON" scripts/physics.py floating --loops 3
"$PYTHON" scripts/physics.py arm --loops 3
```

두 명령 모두 창을 열며, 월드 중력 9.81 m/s²와 접촉을 적용하고 캔은 자유 강체로 둡니다. 플로팅 손만 REGRIND 설정처럼 중력 영향을 끕니다. 팔 결합 모델은 손·팔 모두 중력을 유지합니다. 파지 실패/평가 종료 조건이 발생해도 현재 선택 구간의 전체 5.2초 동작을 끝까지 재생합니다. 각 반복의 시작에서만 로봇과 캔을 초기화합니다. `--headless`는 창 없이 검사할 때만 사용합니다.

플로팅 손목은 `control.py`의 REGRIND 방식 자세 PD를 사용합니다. 위치 오차 × kp − 실제 선속도 × kd의 힘을 root에 적용하고, 회전 오차 × kp − 실제 각속도 × kd의 토크를 전체 링크 질량 비율로 분배합니다. 손의 중력은 `hand_gravity: false`이며 질량·관성과 접촉은 유지합니다. 이전 미리보기 전용 중력 보상과 목표 속도 feedforward는 제거했습니다. 목표는 제어 주기 30Hz로 갱신하고 PD feedback은 물리 주기 120Hz로 계산합니다. 원본 데이터 Z나 PD 게인은 바꾸지 않으며, 몸체 pose는 초기화 때만 대입합니다. 팔 결합 모델은 기존 USD joint-position drive를 사용합니다.

미리보기와 RL은 같은 손목 제어 함수를 사용합니다. RSI·증강·randomization·외란·중력 curriculum은 미리보기에서만 메모리상 OFF로 바꿉니다. GUI에서도 `world.step(render=False)`로 물리를 정확히 한 번 진행한 뒤 별도 `world.render()`로 화면만 갱신합니다. 이전의 마지막 substep `render=True` 호출은 4회 대신 7회 물리를 진행시켜 GUI 동작이 headless와 달라지는 문제가 있었습니다. 플로팅 재생기는 실제 물리 step 수가 제어 decimation과 일치하는지 검사하고 `report.json`에 기록합니다. 종료 자세를 한 제어 주기 더 유지하므로 실제 물리 시간은 5.2333초입니다.

접촉과 남은 추종 오차, 파지 실패는 발생할 수 있습니다. `measurements.npz`의 `timestamp_s`는 reference 종료 시각에 clamp한 비교 시각, `physics_time_s`는 reset 이후 실제 물리 경과 시각이며 `command_time_s`는 해당 제어 주기의 목표 시각입니다. `report.json`에 손목 위치·Z·회전 추종 오차도 기록합니다.

두 모드의 바닥 상단은 `z=0`이고, 각 반복 초기화 직후 실제 캔 collision 밑면 높이를 검사·기록합니다. 기본 팔 재생은 reference의 입력 hash와 배치 행렬도 확인하므로 좌표계 변경 후에는 `scripts/ik.py solve`로 재생성해야 합니다. 캔을 바닥에 고정하는 joint는 추가하지 않습니다.

팔의 `scripts/replay.py --mode targets`도 이제 동적 캔과 충돌 바닥을 사용합니다. Newton mimic은 같은 affine 관계의 PhysX mimic 제약으로 변환하고 종속 관절의 중복 drive만 끕니다. 독립 관절 게인은 유지합니다. 기존 `kinematic` 모드는 기하 검증용으로 유지합니다. 물리 재생은 큰 추종 오차를 측정해도 전체 시연을 완료하면 정상 종료하며, 비유한 물리 상태는 오류로 중단합니다. 결과는 `local/results/physics/floating/`, `local/results/physics/arm/`에 저장합니다.
