# 방법론 출처

FK·기하 리타게팅은 논문과 USD·키포인트 정의로 독립 구현했습니다.
Residual RL은 REGRIND와 로컬 `regrind-revo2`의 방법·설정을 참고했습니다.
FK·RL의 외부 task·환경 소스를 복사하거나 런타임에 import하지 않으며, PPO는 일반 RSL-RL 라이브러리를 사용합니다.

## 논문

| 출처 | 사용한 방법 |
|---|---|
| [REGRIND §3.2, Appendix A](https://arxiv.org/html/2607.11874v1) | semantic correspondence, 손 21점·물체 50점, Delaunay interaction mesh, uniform Laplacian, 순차 최적화 |
| [OmniRetarget §III-A](https://arxiv.org/html/2509.26633v1) | 대응 interaction vertices, Laplacian 제곱오차, configuration smoothness, dt 기반 속도·signed-distance 제약 |

REGRIND의 vertex별 L2 norm 합과 OmniRetarget의 제곱합은 각각 `--norm regrind`, `--norm omni`로 선택합니다.
손·물체 71개 vertex의 residual을 모두 계산하며 프레임 내 connectivity를 고정합니다.

## RL 참고 구성

원본 REGRIND 참고 revision: `0514b6f6501b09cb864e5786063d3293d848c066`.
아래 경로는 원본의 `tasks/manager_based/dexterous/` 기준이며 로봇·object 설정도 함께 참고했습니다.

| 참고 파일·구성 | 적용 내용 | 프로젝트 구현 |
|---|---|---|
| `mdp/actions.py`, `mdp/rb3_revo2_actions.py` | clipped SE(3)/joint residual, 자세 PD, 질량 비례 torque 분배 | `control.py`, `policy/task.py` |
| `mdp/observations.py`, `utils/buffers.py` | 비대칭 actor/critic, 회전 표현, history·noise·delay | `policy/observations.py` |
| `mdp/rewards.py`, `mdp/terminations.py` | 물체 50점·속도·손목·action 보상과 종료 조건 | `policy/task.py` |
| `mdp/commands.py`, `mdp/rb3_revo2_commands.py` | reference-frame RSI, 자세·속도 reset, phase·속도 차분 | `policy/curriculum.py`, `policy/reference.py`, `policy/floating_env.py` |
| `envs/events.py`, `dexterous_env_cfg.py` | 중력 단계, 물성·질량·관성·gain·COM randomization, 외란 | `policy/randomization.py`, `curriculum.py` |
| Revo2 floating config, `free_revo2_right_hand.py`, `tuna_can.py` | 120/30Hz, 손 중력 OFF, 손가락 drive, solver·접촉 설정 | `config/tasks/can_pick/policy*.json`, `policy/floating_env.py` |
| RSL-RL PPO config와 train/play 구성 | PPO·정규화·actor 초기화·checkpoint 흐름 | `policy/ppo.py`, `runner.py` |
| `regrind-revo2`의 접촉 reference 구성 | 물리 rollout의 손–물체 관계를 기준 궤적에 반영 | `policy/rollout_reference.py` |

## 자체 설계

| 기능 | 구현 선택 |
|---|---|
| FK | 양쪽 joint frame과 Newton affine coupling 추출, NumPy FK, semantic 이름 대응 |
| 물체 점 | 삼각형 면적 비례 후보와 farthest-point sampling, 고정 50점 |
| 최적화 | SLSQP, translation+rotation-vector 손목, 이전 성공 해 warm start |
| 추가 추종 | 손 21점 위치와 손가락 segment 방향 항 |
| 충돌 | authored convex hull·원통의 FCL 거리와 보간 표본 제약 |
| 캔 | 사용자 지정 Ø73×32mm, 하단 Ø76×3mm 복합 원통 |
| 좌표 | 캔 밑면 Z=0, 손·물체 공통 변환 |
| RB3 IK | USD 모델, adaptive DLS, branch 연속성, FK 오차·속도·특이점 조건 |
| 상판 보호 | 링크별 collision 경계 상자, 실제/목표 여유 보상, 손목 Z 보정 |
| 접촉 학습 | FK/FCL로 옆면 파지 입력 생성, 패드 접근·대향·동시 접촉 보상, preload |
| 팔 정책 | source-frame 관측, 가상 PD→IK 연결, batched IK를 포함한 팔 학습 |
| 재생 | 시간 재조정, 저장된 12-DoF 궤적 추종, IK 격자 내 회차별 랜덤 배치 |
| 고정 궤적 실행 | 적용된 30Hz 팔·손 명령 기록, 정책 없는 물리 재생, 단위 보정표·초기 자세·지연·feedback 검사 |
| ROS 2 연동 | 단일 장치 I/O 담당, 기록 궤적 action, 측정 12축·모델 종속 관절 구분, 최신 상태의 USD FK 표시 |
| VCB 연결 | 제조사 SDK를 통한 가상 팔 통신, 가상 손 유지, Simulation 모드 검사, 실물과 구분한 ROS namespace·feedback 출처 |

책상·받침대 치수와 위치는 `regrind-revo2/config/workcell/rb3_revo2_table.json`,
일자 손 장착은 원본 어댑터 USD와 attachment 변환을 사용합니다.
자산 출처·hash는 모델과 설정의 metadata에 기록합니다.

## 데이터와 일반 라이브러리

- [DexYCB](dataset.md): 사람 손·물체 annotation.
- [OpenUSD Physics](https://openusd.org/dev/api/usd_physics_page_front.html),
  [Newton USD schema](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/omni.usd.schema.newton/docs/USD_SCHEMAS.html): joint frame·axis·mimic 의미.
- NumPy/SciPy: FK·회전·Delaunay·최적화. [python-fcl](https://github.com/BerkeleyAutomation/python-fcl): 거리·충돌 질의.
- PyTorch/RSL-RL: tensor 계산과 PPO.
- [PhysX compliant contact](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-materials-for-compliant-contacts): 패드 spring/damper 접촉.
- [Physics Tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html): 패드–캔 pair force.
- Three.js/OrbitControls: 비교 뷰어. 배포 번들에 MIT 라이선스를 포함합니다.
- [Rainbow Robotics Servo J](https://rainbowrobotics.github.io/rb_cobot_docs/technical_docs/ui_script):
  관절 degree 명령, 도달/유지 시간과 gain/filter 의미. `hardware.py`에서 공식 `rbpodo` API 사용.
- [BrainCo Revo2 SDK](https://staging.brainco.tech/docs/revolimb-hand/en/revo2/python_sdk.html):
  RS485 연결·0~1000 위치 명령·motor feedback. `hardware.py`에서 공식 `bc-stark-sdk` API 사용.
  실물 관절 변환은 측정 보정표를 요구하며 USD 관절 범위로 SDK 명령값을 추정하지 않습니다.
- [ROS JointState](https://github.com/ros2/common_interfaces/blob/jazzy/sensor_msgs/msg/JointState.msg),
  [FollowJointTrajectory](https://github.com/ros-controls/control_msgs/blob/jazzy/control_msgs/action/FollowJointTrajectory.action): 표준 메시지·action 정의.
  `rclpy` action/QoS API를 사용해 독립 구현했으며 제조사 ROS driver나 예제 server 소스를 복사하지 않습니다.
- [robot_state_publisher](https://github.com/ros/robot_state_publisher/tree/jazzy): URDF·JointState를 TF와 RViz 표시로 연결하는 표준 노드.
  USD 양쪽 joint frame을 보존하는 helper link, instance mesh의 STL 변환, 측정 종속 관절을 유지하는 mimic 없는 표시 URDF는 자체 구현입니다.
  통합 실행기·로컬 조작 패널·quintic 관절 조작·가상 1차 서보·PhysX ROS 장치 adapter도 자체 설계입니다.
  Isaac adapter는 프로젝트의 조립 장면과 관절 position/velocity target API를 재사용하며 정책·IK 실행과 분리합니다.
- [Isaac Sim ROS 설치](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_ros.html): Ubuntu 24.04/Jazzy 환경.
  표시는 직접 `rclpy` 구독과 USD FK를 사용하며 명령 기반 물리 추종과 구분합니다.
- [Rainbow VCB](https://rainbowrobotics.github.io/rb_cobot_docs/technical_docs/virtual_controlbox),
  [TCP 통신](https://rainbowrobotics.github.io/rb_cobot_docs/technical_docs/socket_communication),
  [상태 구조](https://rainbowrobotics.github.io/rb_cobot_docs/technical_docs/data_structure):
  VM 설치·Simulation 전용 모드, 5000/5001 포트, `jnt_ref`/`jnt_ang` 의미와 packed fault bit.
  공식 `rbpodo==0.16.14`를 사용합니다. 가상 손·초기 자세 Move J·30Hz ROS 연결·지연 검사는 자체 설계이며
  Servo J 시험값은 제조사 권장 튜닝값으로 간주하지 않습니다.

## 작업 구성

Task registry, 작업별 설정·데모·정책 분리, `drilling`의 입력/단계 정의는 이 프로젝트의 자체 설계입니다.
공통 리타게팅·PPO와 작업별 환경·보상을 분리합니다. 드릴 입력과 단계 구성은 독립적으로 정의했습니다.

## RGB 데이터 제작

| 출처 | 사용 방식 |
|---|---|
| [EgoPHI 프로젝트](https://siplab.org/projects/EgoPHI), [공식 코드](https://github.com/eth-siplab/EgoPHI) | 외부 저장소의 `InteractionGNN`과 공식 checkpoint를 별도 Python 프로세스에서 호출. mesh 정규화·graph·RGB 전처리는 공식 추론 계약을 참고 |
| [HaMeR 공식 코드](https://github.com/geopavlakos/hamer) | 외부 `load_hamer`, `ViTDetDataset`, `cam_crop_to_full` API로 손 mesh·21점 추정. 필요한 MANO 모델은 해당 이용 조건에 따라 별도 준비 |
| [OpenCV 카메라 보정](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), [ArUco](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html) | checkerboard 보정·측정한 marker corner의 PnP·재투영 오차 계산 |

원본 신경망 코드를 프로젝트에 복사하지 않습니다. 저장소·가중치는 `local/`에 두고 모델 파일과 checkpoint의 hash를 출력 metadata에 남깁니다.
HaMeR의 단안 절대 위치는 `monocular_estimate`로 표시합니다. EgoPHI 데이터 로더의 정답 손 위치 보정은 사용하지 않으며, 선택적 metric 정렬에는 별도로 측정한 손 관절 anchor가 필요합니다.

촬영 시간 보존, 입력 계약, 측정한 mesh·책상 좌표, marker 기반 물체 pose, 누락 프레임 처리, 검수 화면, task별 내보내기는 자체 설계입니다.
EgoPHI의 물체 추정은 독립 추적 pose와 일관성을 검사해 별도로 저장하며 원본 추적 궤적을 덮어쓰지 않습니다.
양손 문맥이 없는 프레임은 기본적으로 접촉 추론을 건너뜁니다. `zero_context_debug`는 유효 라벨로 인정하지 않는 진단용 입력 방식입니다.
힘은 `normalized_model_output`으로 저장하며 드릴에서 측정·보정한 Newton 값, 마찰력 또는 토크로 취급하지 않습니다.
공식 checkpoint 전체를 strict load하며, 중복 다운로드를 피하도록 모델 생성 중에만 별도 backbone 사전학습 가중치 로딩을 끕니다. 가정한 보정값·관절 대용점을 쓰는 공개 샘플 시험은 `test_assumptions`로 분리합니다.

## Revo2 tactile

[BrainCo Revo2 Touch 프로토콜](https://staging.brainco.tech/docs/revolimb-hand/en/revo2/modbus_touch.html)의
정상력·접선력 0.01 N/count, 0–25 N 범위, 방향각·status 의미를 사용합니다.
공식 SDK의 `get_touch_sensor_status()`를 읽기 전용으로 호출하며 원본 예제 소스는 복사하지 않습니다.
PhysX 접촉점 힘·마찰력 합산, 120 Hz 기록과 USD 패드 축 유도는 자체 구현입니다.
센서 통신 형식 근사와 실물 센서 보정을 구분하며, ADC·근접 응답·노이즈·대역폭은 모델링하지 않습니다.

동작 중 촉각 기록은 공식 `get_touch_sensor_status` API와 위 통신 단위를 사용합니다.
단일 RS485 소유자, 명령 사이 여유 시간 폴링, 공통 monotonic 시각, 고정 명령 SHA-256 정렬,
실측 관절·촉각 CSV 및 정상력/접선력/방향각 비교 그래프는 자체 설계입니다.
제조사 내부 신호처리나 노이즈 모델을 복제하지 않습니다.

## 패드 눌림 근사

[PhysX compliant contacts](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#compliant-contacts)의
접촉점별 implicit spring–damper를 사용합니다.
다섯 패드의 강성 10,000 N/m·감쇠 20 N·s/m는 사용자 요청의 약 1~2mm 눌림을 위한 자체 설정이며 실측 고무 물성이 아닙니다.
[접촉·rest offset](https://nvidia-omniverse.github.io/PhysX/physx/5.1.3/docs/AdvancedCollisionDetection.html#tuning-shape-collision-behavior)에 따라
rest offset은 0으로 유지합니다. 관통 깊이는 PhysX가 보고하는 접촉 간격으로 계산하고 2mm 초과도 그대로 기록합니다.

## 손가락 목표각 추종

실측 관절각 기준 목표각 제한, USD 속도 한계와의 교집합 처리, 제한 전 명령·제한 후 실제 추종 오차 보상은 자체 설계입니다.
작은 목표각 차이에서도 PD 파지 토크를 유지하도록 이 설정의 강성·감쇠를 조정하되 기존 토크 상한을 유지합니다. 도달 불가능한 제한 구간은 별도 기록합니다.
기존 actor 평균·정규화 통계만 이관하고 critic·optimizer를 새로 시작합니다. 실물 관절 보정이나 힘 제어를 대신하지 않습니다.

## 접촉점 리타게팅

| 출처 | 참고한 방법 | 독립 구현·차이 |
|---|---|---|
| [C2Dex §III](https://arxiv.org/pdf/2608.07045v2) | object-local 접촉점 군집·medoid, semantic 접촉 대응, interaction mesh | 논문은 접촉 **손실**을 사용한다. `contact_retargeting.py`는 접촉점–패드 표면 거리를 hard inequality로 제한한다. MANO 피부 mesh·silhouette 대신 현재 캔 mesh에 투영한 21점 skeleton tip을 사용하므로 추정 접촉점이다. |
| [SPIDER §2.2–2.3](https://arxiv.org/html/2511.09484v1#S2.SS3) | simulator-in-the-loop 궤적 샘플링, Boltzmann 가중 갱신, 접촉 쌍의 가상 힘과 점진적 제거, 짧거나 이동이 큰 접촉 제외 | `contact_physics.py`는 PhysX의 실제 패드·캔에 반대 방향의 spring/central-damper 힘과 COM 기준 torque를 가한다. 100→50→20→0% 보조 후 반드시 0 N 재생한다. 원본 소스는 사용하지 않는다. |

접촉 구간 26–41번은 사용자 지정이다. 두 방법은 동일한 고정 접촉점을 공유할 수 있다.
`human_surface_proxy`는 사람 손끝 투영점을 유지한다. `robot_surface_seed`는 기존 유효 로봇 궤적의 캔 표면 접촉점으로 위치를 재선정하는 자체 형상 적응이며, 원래 점과 이동량을 별도로 저장한다.

Hard 방식은 0.75 mm 이내의 **표면 접촉**이며 패드 위 접촉 위치는 이동 가능하다. 고정된 로봇 local point 5개의 15차원 equality weld를 의미하지 않는다.
전체 손 collision mesh의 비관통·관절/속도 한계·보간 표본을 별도로 검사하며, 목적함수 수렴 실패 시 입력 주변의 제약 복원 여부를 기록한다.

SPIDER에서 PhysX, 현재 Revo2 PD·측정각 기반 governor, 12개 control knot, 64개 후보, 자체 tracking/contact 비용, 손가락 동시 굽힘 초기 후보는 자체 선택이다.
분산은 본문의 coarse-to-fine 설명에 따라 감소시킨다. Eq. (3)의 인쇄된 지수식을 그대로 쓰지 않는다.
worst-case 물성 최적화·정책 증류는 포함하지 않는다. 가상 힘이 있는 rollout과 무보조 결과를 구분하며, 접촉력·물체 오차·coupling 오차를 각각 기록한다.
