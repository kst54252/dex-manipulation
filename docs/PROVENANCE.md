# 구현 출처와 설계 선택

이 문서의 FK·기하학적 retargeting 단계는 지정 논문, 현재 저장소의 USD·키포인트 JSON·데이터 배열, 일반 라이브러리 API를 바탕으로 새로 구현했다.
그 단계에서는 원본 REGRIND/OmniRetarget 구현 및 외부 `regrind-upload`/`regrind-revo2` 소스를 열거나 복사하지 않았다.
이후 사용자가 별도로 요청한 **residual RL 단계**에서는 `/home/wanjunkim/regrind`의 코드를 읽고 방법을 참고했다.
RL의 참고 파일·채택한 개념·독자 구현 범위는 [아래 RL 출처 표](#residual-rl-출처와-독립-구현)에 따로 기록한다. 앞선 FK/retargeting 코드를 외부 구현으로 교체하지 않았다.
기존 에셋 설명에 적힌 값도 그대로 코드에 옮기지 않고 composed USD를 읽어 확인했다.

## 논문에서 채택한 방법

| 출처 | 채택한 내용 |
| --- | --- |
| [REGRIND §3.2, Appendix A](https://arxiv.org/html/2607.11874v1) | 사람/로봇 손의 semantic correspondence, 손 21점과 물체 표면 50점, Delaunay interaction mesh, uniform Laplacian 변형오차, 순차 최적화와 이전 해 warm start, 물체 pose 고정, floating wrist와 손 관절 최적화, 관절·속도·비관통 제약 |
| [OmniRetarget §III-A](https://arxiv.org/html/2509.26633v1) | source/target에 대응하는 interaction vertices, 균등 이웃 가중치, Laplacian 제곱오차 대안, configuration smoothness, dt에 비례한 속도 제한, collision-pair signed distance 제약 |

REGRIND 식 (4)는 vertex별 L2 norm의 **합**, OmniRetarget 식 (2)는 **제곱합**이다.
이를 같다고 처리하지 않는다. 기본 `--norm regrind`는 전자, `--norm omni`는 후자를 선택한다.
손과 물체를 모두 포함한 71행의 residual을 사용한다. 물체 좌표가 고정이어도 손에 연결된 물체 vertex의 Laplacian residual은 0이 아닐 수 있다.

## 자체 설계와 논문 대비 차이

- 2026-09-20 사용자 치수에 따른 캔 교체는 자체 설계(`object.py`, `assets/can/spec.json`)다. 전체 높이 32 mm 중 하단 3 mm는 Ø76 mm, 나머지 29 mm는 Ø73 mm다. 원래 collision cylinder의 축·기하 중심을 보존해 원본 6D pose와 사람 손 annotation을 수정하지 않는다. 캔 원본은 `assets/can/original/`, 이전 결과/설정은 `local/archive/before_can_resize/`에 보관한다. 원본 영상에 대한 기존 IoU는 새 형상의 정확도를 뜻하지 않는다.
- 교체 캔의 시각 메시에는 바닥·하단 외벽·고리 모양 턱 윗면·몸통·윗면을 포함하고 내부 접합면은 넣지 않는다. 원주 분할 256은 자체 선택이며 collision은 정확한 두 원통의 합집합으로 검사한다. 표면 50점은 기존 면적 샘플링/FPS 방법으로 새로 생성했다. 0.15 kg은 기존 시뮬레이션 값을 유지한 것이며, COM/관성은 균일한 고체 밀도 가정이지 실측값이 아니다. 두 collider를 모두 사용하는 바닥 지지 높이와 geometry fingerprint/asset hash 검사도 자체 설계다. [치수·파일·가정](object.md)에 자세히 기록했다.

- 바닥 좌표계 변환(`coordinates.py`, `scripts/ground.py`)은 사용자 요청에 따른 자체 설계다. 실제 캔 collision cylinder의 오프셋·크기를 사용해 첫 캔의 local +Z를 위로 세우고 밑면을 z=0에 둔다. 모든 손/물체 pose·키포인트에 같은 SE(3)를 적용하고 joint/time/local geometry는 보존한다. float32 회전의 반올림 오차를 제거한 회전으로 좌표변환을 만들되 개별 source pose를 보정하지 않는다. 원본 카메라 데이터는 유지한다. RB3용 추가 변환은 Z축 −90° 회전 후 베이스 X방향 0.3 m 이동이며 실측 보정이나 논문의 테이블 복원 방법을 뜻하지 않는다.
- 물리 동작 미리보기는 후속 사용자 요청에 따라 아래의 로컬 REGRIND 손목 제어 방법과 동일한 함수를 사용한다. 이전 자체 설계의 root 중력 힘/모멘트 보상·reference 속도 feedforward는 제거하고 Git 제외 영역 `local/archive/before_floating_gravity_fix/`에 보관했다. 데이터/FK/기하 retargeting 구현의 출처는 바뀌지 않는다.
- 회전된 팔 궤적에는 원본 시각·자세를 정확히 유지한 채 구간당 중간 표본 하나를 추가한다. 위치/finger 선형 보간과 회전 SLERP는 자체 선택이다. 20° step·실제 dt 속도·특이점 기준을 완화하지 않으며, 추가 표본은 원본 camera frame으로 표시하지 않는다.
- USD 추출 결과를 JSON/NPZ로 분리했다. 실행 FK는 NumPy/SciPy만 사용하며 USD·Isaac를 import하지 않는다.
- FK는 column-vector 규약으로 `T_child = T_parent F0 motion(q) inverse(F1)`를 적용한다. body 관계를 역방향으로 통과하면 상대변환 전체를 역변환한다. 양쪽 joint frame을 모두 사용한다.
- USD의 **NewtonMimicAPI** affine coupling을 읽는다. 실제 독립 관절 수는 USD 해석 결과 6, 가동 관절 수는 11이다. 전체 관절의 한계·속도 제한을 coupling으로 독립 좌표에 환산한다. 미지원 joint/collider/coupling 형식은 오류로 중단한다.
- composed PhysX 속도 제한을 우선한다. `urdf:limit:velocity`는 별도 원본 메타데이터로 기록한다. 검지 proximal은 USD에서 90°/s로 덮어써져 있으며 기존 URDF 값과 다르다.
- 키포인트 JSON의 이름에서 semantic 이름을 얻고 DexYCB의 명시적 관절 이름과 연결한다. 번호끼리 직접 대응하지 않는다. JSON의 parent-link local 좌표를 USD의 키포인트 Xform과 대조한다.
- 캔의 실제 메시 삼각형에서 면적 비례로 후보 20,000점을 뽑고 farthest-point thinning으로 50점을 선택한다. seed는 20260918이다. 삼각형 번호와 barycentric 좌표도 저장하므로 표면 소속을 재검증할 수 있다. 표본은 한 번만 생성한다.
- 각 source 프레임에서 그래프를 한 번 계산하고 그 프레임의 모든 optimizer 평가에 같은 연결을 사용한다. Qhull 실패/누락 vertex는 실패로 표시하고 그래프를 몰래 변경하지 않는다.
- MOSEK/Drake의 별도 QP/SOCP 대신 SciPy SLSQP를 사용한다. REGRIND norm의 원점은 `sqrt(r² + 1e-14) - 1e-7`로 매끄럽게 만든다. 일반 라이브러리 외의 최적화 구현은 사용하지 않았다.
- 손목 변수는 translation 3개와 이전 손목 회전에 대한 rotation-vector 3개다. SO(3) exponential로 회전행렬을 만들고 저장할 때 XYZW quaternion도 제공한다. quaternion component에 직접 bound를 두는 논문 구현과 다르다.
- 첫 프레임은 wrist와 네 손가락 MCP의 proper rigid alignment로 초기화한다. 이후는 이전 **성공한** 해를 사용한다. 실패 해를 정상 해로 재사용하지 않는다. 실패 뒤에는 이전 성공 프레임과의 실제 시간차를 사용하고 인접 transition mask를 별도로 저장한다.
- smoothness는 이전 해와의 translation/0.1m, rotation-vector(rad), 독립 관절(rad) 차이의 제곱합에 0.02를 곱한다. 물체 vertex는 smoothness 변수가 아니다.
- 원본과의 동작 차이를 줄이기 위해 **자체 추종 항**을 추가했다: `200 × sum_21 ||p_robot-p_human||²`와 `0.5 × sum_15 ||u_robot-u_human||²`. `u`는 각 손가락 MCP→PIP→DIP→tip의 단위 방향이다. 원본 손/물체 좌표나 interaction graph를 스케일링·변형하지 않으며, Laplacian과 smoothness 항도 유지한다. 이 두 항은 논문에서 가져온 것이 아니다. `--position-weight 0 --direction-weight 0`으로 기존 논문 목적함수 구성을 재현한다. 방향만 강조한 시험은 위치·fingertip 오차가 커져 채택하지 않았다.
- `valid`는 solver/제약 검사 통과일 뿐 데이터셋 동작과의 일치를 뜻하지 않는다. 위치 RMSE·손목·손끝·15개 손가락 방향·손바닥 회전 오차를 별도로 기록한다. 현재 Revo2의 6개 독립 관절과 실제 coupling은 인간의 모든 손가락 굽힘을 표현하지 못한다. 따라서 추가 항으로 완전한 동작 일치를 주장하지 않는다.
- floating wrist에는 제조사 제한이 없으므로 **알고리즘 설정**으로 선속도 1m/s, 각속도 8rad/s를 둔다. CLI에서 명시적으로 바꿀 수 있다. 손 관절 속도는 USD에서 읽는다. 이 값들은 하드웨어 안전 사양이 아니다.
- collision margin은 0.5mm, 제약 허용 오차는 2e-6, SLSQP 최대 반복은 시도당 400, ftol은 1e-6이다. 실패하면 같은 후보에서 Hessian 근사를 초기화해 최대 3회 시도하고 각 시도 상태를 기록한다. 실패 해를 성공으로 바꾸거나 제약 허용치를 늘리지 않는다.
- 접촉점 부근 GJK 거리의 미소 차분 잡음을 피하려고 FCL 최근접점과 법선을 이용한 signed-distance 미분을 구현했다. 최근접점을 link-local로 고정한 뒤 FK의 중앙 차분으로 점의 Jacobian을 얻는다. objective의 수치 차분 간격은 1e-6이다. 기존 ftol=1e-8은 접촉 거리 질의의 정밀도에 비해 과도해 실제 데이터에서 수렴 실패가 있었고, 1e-6으로 조정했다. 최종 geometry/속도/관절 검사는 따로 수행한다.
- 추가 위치 항이 매우 큰 불연속 목표에서는 objective를 프레임 시작 시 고정한 `max(1, initial_objective/10)`으로 나눠 수치 conditioning을 조절한다. 양의 상수라 최적해를 바꾸지 않는다. 원래 에너지로 보고하고 제약 허용치는 바꾸지 않는다. 현재 실제 시퀀스의 scale은 모든 프레임에서 1이다.
- 프레임 사이 재생에서 translation·관절값은 선형, 회전은 SLERP로 보간한다. 원본 두 자세가 충돌하지 않아도 그 사이가 관통할 수 있어, 각 구간의 alpha=0.1,…,1.0에 동일한 **전체 collision geometry** 제약을 추가했다. 중간 물체 pose는 고정된 보간이며 최적화 변수가 아니다. 이 표본 제약은 논문에서 그대로 옮긴 것이 아닌 재생 오류 수정이다. 별도 100등분 검사는 연속 시간의 엄밀한 충돌 증명이 아니다.
- 손 충돌 메시의 `convexHull`을 SciPy ConvexHull로 구성하고 FCL의 GJK/EPA 거리 질의를 사용한다. 키포인트나 vertex 샘플만으로 충돌을 판단하지 않는다. 캔은 USD의 모든 collision cylinder를 사용한다(현재 하단 턱/몸통 2개). 최종 상태는 별도의 FCL collide 질의로도 확인한다.
- 이 충돌 검증은 **프레임과 유한한 보간 표본의 손–캔 collision geometry** 범위다. 연속 시간 swept collision, 손 자기 충돌, 테이블/환경, PhysX contact/rest offset, 마찰·힘·물리적 파지 성공은 검증하지 않는다. PhysX가 내부적으로 단순화한 hull과 FCL의 원본 전체 hull이 비트 단위로 같다는 주장도 하지 않는다.
- 누락된 metadata를 임의로 채우지 않는다. 실제 시퀀스에서는 `joint_3d/joint_2d`로 독립 추정한 intrinsics와 36장 segmentation의 메시 투영을 비교해 object index=0, mesh→object=identity를 채택했다(mean IoU 0.9535). 오른손은 RGB 15/33/50에서 관찰했다. 이들은 원본 metadata를 확보했다는 의미가 아닌 관찰·재투영 근거의 판단이며, 축 대칭 캔의 완전한 orientation 식별을 주장하지 않는다.
- 사용자가 느린 시간 배정을 허용했으므로 `time_basis=retimed_user_authorized`, `retimed_fps=5`를 명시해 0.2초 간격을 정의했다. 원래 `fps`는 null이며 촬영 속도를 추정한 값이 아니다. 속도 제약은 이 새 실행 시간에 대해 검사한다. 시간/충돌 검증을 생략하는 호출은 `geometric_debug`로 분류하며 모든 `valid`를 false로 저장한다.
- Three.js 뷰어는 정적 메시를 한 번만 저장하고 손목/관절/물체 pose만 갱신한다. 브라우저에서도 양쪽 joint frame과 coupling을 적용해 FK를 계산한다. 이전 Plotly 뷰어의 프레임별 전체 mesh 복제를 제거했다. Three.js와 OrbitControls는 일반 시각화 의존성이며 번들과 MIT 라이선스를 함께 배포한다.
- 실제 데이터 뷰어에는 원본 JPEG와 `joint_2d`를 포함한다. `joint_3d→joint_2d`에서 복원한 pinhole camera로 세 화면의 시점을 맞추고, 기본은 보간 없이 동일 frame ID를 표시한다. 보간은 명시적 선택이며 RGB를 새로운 촬영 프레임처럼 보간하지 않는다. 원본 calibration 파일을 확보한 것이 아니라 제공 label의 재투영을 수치 검증한 것이다.

## 데이터·API 근거

후속 residual RL은 사용자가 로컬 REGRIND 방법론 참조를 별도로 허용한 단계다. 참고한 파일·일반 RSL-RL 의존성과 자체 개선은 [아래 RL 출처 표](#residual-rl-출처와-독립-구현), 현재 원본 대비 차이는 로컬 대조표 `local/reports/RL_DIFFERENCES.md`에 구분했다. 이 허용을 앞선 retargeting 코드의 소스 참조로 소급하지 않았다.

- [DexYCB 공식 label/관절 정의](https://github.com/NVlabs/dex-ycb-toolkit#loading-dataset-and-visualizing-samples): `pose_y`, `joint_3d`, 순차 MANO21 이름. toolkit 구현을 retargeting 코드로 사용하지 않는다.
- [OpenUSD Physics](https://openusd.org/dev/api/usd_physics_page_front.html): body 관계, local joint frames, axis와 각도 단위.
- [NVIDIA Newton USD schema](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/omni.usd.schema.newton/docs/USD_SCHEMAS.html): follower = coef0 + coef1 × leader, angular offset 단위 degree. 생략된 coef0/coef1의 기본값 0/1을 사용한다.
- [python-fcl](https://github.com/BerkeleyAutomation/python-fcl): 일반 collision library의 Convex/Cylinder/Transform와 거리·충돌 API.
- 현재 자산 경로, 수치, source hash는 `assets/models/`에 기록한다. 입력 무변경 검사는 `local/results/validation/input_hashes.json`을 사용한다.

## Residual RL 출처와 독립 구현

2026-09-18 사용자 요청으로 로컬 `/home/wanjunkim/regrind`의 RL 방법론을 참고했다. HEAD는 `0514b6f6501b09cb864e5786063d3293d848c066`이며 Isaac Lab 3/Isaac Sim 6 대응판이다. 읽은 파일의 해시는 로컬 manifest `local/archive/before_can_resize/local/results/policy/source_provenance.json`에 기록한다.

**REGRIND의 함수·클래스·환경 파일을 복사하거나 런타임 import하지 않았다.** 현재 PPO는 원본도 쓰는 **일반 RSL-RL 5.4.1 라이브러리**를 정상 import한다. task, 관측 버퍼, reference, curriculum, PhysX adapter 및 저장·평가 orchestration은 이 저장소의 독립 구현이다. “PPO까지 전부 자체 구현”이라는 이전 설명은 현재 버전에 해당하지 않는다.

### 방법론 대응표

원본 경로 prefix는 `/home/wanjunkim/regrind/source/regrind/regrind/`; 아래 mdp/config는 `tasks/manager_based/dexterous/` 아래다.

| 참고 파일·방법 | 채택한 내용 | 현재 파일 |
|---|---|---|
| mdp/actions.py, relative joint / SE3 impedance | clip residual + reference, 위치/회전 PD, 질량 비례 분산 torque | policy/task.py, env.py, control.py |
| robots/free_leap_right_hand.py, Leap screwdriver PLAY config | 손의 전체 rigid body 중력 OFF, 재생 시 월드/물체 중력 9.81 m/s² | config/policy.json, policy/env.py, sim.py |
| mdp/observations.py, dexterous_env_cfg.py | 비대칭 actor/critic, 6D 회전, history2, noise·연속 delay, default q offset | policy/observations.py, math3d.py |
| utils/buffers.py, envs/events.py | 그룹별 연속 lag 및 quaternion SLERP의 의미 | 독립 고정 길이 GPU 버퍼; 해당 클래스 사용 안 함 |
| mdp/rewards.py, Leap config | 50점 평균 거리, 물체 선/각속도, 손목, raw action 변화·범위 exponential reward, dt 적분 | policy/task.py |
| mdp/terminations.py | object deviation, workspace, 손–물체 거리, terminal demo end | policy/task.py, env.py |
| mdp/commands.py | 정수 reference frame RSI, 실패 histogram/EMA, 시연 상태·속도 reset | policy/curriculum.py, env.py |
| mdp/commands.py, Leap rigid config | episode XY/yaw 증강 및 선형 fade phase .135→.292 | policy/reference.py; 로컬 검산용 `local/tests/helpers/augmentation.py` |
| envs/events.py, dexterous_env_cfg.py | 13단계 중력 범위, reset event, interval velocity push | policy/curriculum.py, randomization.py |
| dexterous_env_cfg.py, robot/task config | 재질 bucket, 질량·관성·gain·default q·COM randomization | policy/randomization.py |
| config/leaphand/agents/rsl_rl_ppo_cfg.py, utils/rl_cfg.py | PPO hyperparameters·network·normalization | config/policy.json, policy/ppo.py |
| modules/actor_critic.py | 마지막 actor layer를 0으로 초기화하는 아이디어 | 일반 torch API로 마지막 Linear만 zero |
| scripts/rsl_rl/train.py, play.py | 초기 episode length 분산, source PLAY 조건, checkpoint/export | policy/runner.py, scripts/policy.py |
| Leap rigid screwdriver config | 9초, workspace, augmentation phase, COM 범위 | rigid can에 동일 recipe 사용; 데이터/형상은 기존 can |

### 플로팅 물리 재생: 2026-09-20 수정

사용자의 떨림 보고 후 같은 HEAD의 `robots/free_leap_right_hand.py`, `mdp/actions.py`, `config/leaphand/leap_base_env_cfg.py`, `config/leaphand/leap_screwdriver_env_cfg.py`를 다시 확인했다. 채택한 방법은 다음과 같다.

- 손의 모든 rigid body에 `disable_gravity=True`; 캔은 중력·접촉을 받는 자유 강체다. 미리보기에서 `hand_gravity=True`를 강제로 덮어쓰던 코드를 제거했다.
- `F = kp_pos (p_target − p) − kd_pos v`, `τ = kp_rot Log(R_target Rᵀ) − kd_rot ω`. 힘은 root COM에 적용하고 토크는 `m_i / Σm` 비율로 전체 링크에 나눈다. 중력 보상이나 목표 속도 feedforward는 사용하지 않는다. 수식의 독립 구현을 `control.py`에 두고 미리보기와 RL이 함께 호출한다.
- 기존 kp/kd `(300, 30, 3, 0.3)`, 120Hz 물리/30Hz 목표 갱신을 유지한다. REGRIND Leap 전용 손가락 gain·질량·관성을 Revo2에 덮어쓰지 않는다. Revo2의 실제 coupling·joint limits·6개 독립 관절 adapter는 유지한다.

**환경 adapter의 버그 수정:** 설치된 Isaac Sim `SimulationContext.step/render`와 Isaac Lab `manager_based_rl_env.py`의 실제 실행 순서를 확인했다. 이전 `World.step(render=True)`는 rendering dt=1/30 안에서 물리를 4번 진행하여 앞선 명시적 3회와 합쳐 제어당 7회가 됐다. 원본 Isaac Lab처럼 항상 `step(render=False)`로 4회 진행하고 `render()`로 화면만 갱신하도록 수정했다. 제어기를 바꾸지 않아도 필요한 GUI 시간 진행 수정이며, 새로운 RL 방법을 도입한 것이 아니다. 재생 중 실제 물리 step 수를 검사하는 guard와 `physics_time_s` 기록은 자체 검증 설계다.

원본 함수·클래스 복사나 런타임 의존은 추가하지 않았다. 비교 실험·120Hz 기록·원본 파일 hash·테스트 로그는 Git 제외 영역 `local/reports/floating_gravity/`에 보관한다. 관절/손목 진동과 목표 추종 지연을 구분하며, 무진동이나 파지 성공을 가정하지 않는다.

### 일반 라이브러리와 자체 선택

- RSL-RL 5.4.1의 PPO, MLPModel, GaussianDistribution, RolloutStorage를 **의존성으로 사용**한다. PPO loss/optimizer/normalizer/분포 본문을 가져다 붙이지 않는다. 일반 Isaac Lab의 RewardManager, reset/step 순서, mass/material randomizer, joint action 및 설치된 PhysX tensor API도 동작 확인에 사용했다.
- PyTorch, TensorDict, NumPy, SciPy, OpenUSD, Isaac Sim의 공개 API 사용. JIT/ONNX export도 일반 라이브러리 API다.
- **유지한 자체 개선:** final-observation timeout bootstrap, pose의 분석적 속도 미분, reference/model/config contract 검사, curriculum/물성/RNG 저장, 원본 PLAY 외의 엄격한 추종 평가. 학습 성능 우월성은 측정하지 않았다.
- **로봇/데이터 adapter:** NPZ joint/semantic 순서 검증, invalid reference 거부, XYZW 내부 quaternion과 Isaac 경계 WXYZ 변환, Revo2 독립6/전체11 coupling·실제 limits·target slew, COM/origin velocity 변환, collision geometry와 실제 link pose 기록.
- Newton mimic 의미를 PhysX mimic 관계로 바꾸는 코드는 직접 작성했다. SDK 관계 `q_f + G q_l + offset = 0`에 따라 `G=-multiplier`, `offset=-degrees(offset_rad)`를 author한다. source asset은 수정하지 않는다.
- 캔 50점은 기존 retargeting에서 확인한 실제 mesh/geometry와 고정 index를 사용한다. RL에서 6D pose로 형상을 추정하지 않는다.
- initial can +Z를 가상 table up으로 쓰는 frame은 실측 calibration이 아니다. 원 촬영 fps는 미확인이고 RL만 9초 horizon으로 retime한다. 사용자 요청으로 원본 42~50번을 제외해 현재 기하 궤적은 15~41번의 27자세, 5.2초다. 남은 자세와 0.2초 간격은 바꾸지 않는다. 잘라내기 전 원본은 `local/archive/before_tail_trim/`에 보관한다.
- REGRIND retargeter, OmniRetarget 구현, 기존 regrind-revo2 코드는 참고·복사하지 않았다. 앞선 retargeting 출처는 [PROVENANCE.md](PROVENANCE.md)에 유지한다.

원본 저장소의 MIT 및 일부 모듈 BSD-3-Clause 표시를 확인했으며 원본 코드는 재배포하지 않는다. 일반 패키지는 설치본의 라이선스를 따른다. 상세 차이와 미이식 선택 기능까지 로컬 전체 대조표 `local/reports/RL_DIFFERENCES.md`에 공개한다. 이번 정렬 과정에는 **실제 정책 학습을 실행하지 않았다**.


## RB3 arm IK: 독립 수치 구현

이 단계는 REGRIND/OmniRetarget 코드나 외부 RB3 IK 구현을 복사하지 않고 현재 조립 USD와 일반 NumPy/SciPy/Isaac API로 구현했다. 논문의 retargeting 목적함수를 arm IK 알고리즘으로 사용하지 않았다.

- 실제 composed USD의 joint graph, 양쪽 frame, axis, limits, 속도와 assembler attachment를 추출했다. `link6`를 USD flange-link frame으로 명시하고 제조사 flange face/TCP offset을 추정하지 않았다.
- analytic spatial Jacobian, SO(3) rotation-vector 오차, bounded adaptive damped least squares, joint-limit regularization, 결정적인 multi-start와 이전 해 거리 선택은 이번 독립 설계다. strict FK pose 판정은 optimizer 종료 상태와 별개다.
- SVD 길이 정규화 0.5 m, pose tolerance 1e-5 m/1e-4 rad, warning/failure σ .03/.001, 20° frame step, 20등분 transition 검사는 하드웨어 사양이 아닌 명시적인 알고리즘 설정이다.
- 카메라–RB3 실측 보정이 없어 시뮬 배치를 별도 JSON으로 만들었다. 첫 wrist와 선택한 굽힌 팔 seed를 연결하는 단일 강체 변환으로 모든 source pose/물체를 옮겼다. 원본 궤적·손가락·시간을 수정하지 않았다. 실패한 첫 배치 결과도 로컬 기록에 남겼다.
- Pure `JointReference`와 Isaac adapter를 분리했다. 상태 재생의 link FK 비교를 동적 추종이나 실제 제어 성공으로 보고하지 않는다. arm/environment collision, hardware calibration, acceleration/jerk shaping은 미검증/미구현이다.
# Workcell dimensions and placement (2026-09-20)

- 사용자 요청으로 `regrind-revo2/config/workcell/rb3_revo2_table.json`의
  로봇 받침대·책상·다리 치수/위치, floor 높이, robot mount pose만 참고했다.
  `docs/architecture.md`의 workcell 계약과 `docs/ISAAC_SIM_REPLAY.md`의 좌표 설명으로 확인했다.
  출처 hash는 `config/workcell.json`에 기록했다. 이 작업에서는 해당 저장소의 실행 코드를 복사하거나 import하지 않았다.
- 자체 구현: `scene.py`의 USD box 생성, 조립 USD의 static anchor를 포함한 장착 변환,
  tabletop world↔RB3 base 변환, reference fingerprint 검증, simulator 경계의 floating XY 평행이동.
  기존 PD·RL·IK 알고리즘/게인은 유지했다. 환경 간격 2.2m, 표시용 바닥 크기/색상/카메라 시점은 자체 선택이다.
- 책상 위 z=0, 로봇 장착 z=-0.02m, 바닥 z=-0.72m. 원본 asset/data는 수정하지 않았다.
  상세 수치와 사용법은 [scene.md](scene.md)에 있다.

## 2000 iteration 실험용 중력 스케줄 (2026-09-20)

- 사용자 요청의 4096 환경 / 2000 iteration 실험에 맞춰 중력 단계 시작 시점만 단축했다.
  PPO 24 control steps/iteration 기준, 400 iteration부터 중력을 올리고 이후 100 iteration마다
  원래의 다음 중력 범위를 적용하며 1500 iteration(36000 control steps)부터 9.81 m/s²로 고정한다.
  13단계 범위와 reset 시 적용 방식은 원본 방법을 유지한다. 기본 학습 길이는 2000 iteration으로 변경했다.
  이는 사용자 요청의 스케줄 변경이며 원본 130000-step 스케줄과 구분한다.

## 플로팅 전용 평면 (2026-09-20)

사용자 요청으로 플로팅 학습·평가·물리 재생의 큰 책상/받침대를 20×20cm 지지면으로 교체했다.
후속 사용자 요청으로 지지면 크기만 **30×30cm**로 확대했다.
`FloatingSurface`는 별도 외부 소스를 참고하지 않은 자체 구현이다. 두께 2cm의 유한 box collision,
윗면 z=0, 첫 캔 기준 위치를 평면 중앙으로 맞추는 공통 XY 이동과 환경 간격 0.75m를 사용한다.
기존 증강·중력 curriculum·제어/보상은 유지한다. 팔 환경의 source workcell 배치와 IK는 변경하지 않았다.

## 4096 환경 초기화 (2026-09-20)

동일 손·캔 USD를 환경마다 다시 구성하던 부분을 source 환경 1개 + 일반 Isaac Sim
`Cloner.clone(replicate_physics=True, enable_env_ids=True)`로 교체했다.
설치된 Isaac Sim 6.0.1의 `isaacsim.core.cloner/impl/cloner.py` API와
`isaacsim.core.prims`의 생성자/xform 계약을 확인했으며 외부 REGRIND 코드는 참고하지 않았다.
초기 pose를 복제 전에 한 번 설정하고 view 생성 때 중복 USD xform 변경을 생략한다.
관절 coupling·제어·보상·중력 스케줄·증강은 유지하며 초기화 단계별 진행 로그를 자체 추가했다.

## Iteration 통계와 성능 비교 (2026-09-20)

사용자 요청으로 `regrind-revo2`의 floating Revo2 RL 환경·agent 설정
(`revo2_floating_tuna_env_cfg.py`, `agents/rsl_rl_ppo_cfg.py`, `robots/free_revo2_right_hand.py`)을
읽어 물리 dt, decimation, PPO batch/모델 크기, solver와 접촉 설정을 대조했다.
해당 저장소의 FK/retargeting 코드는 읽거나 가져오지 않았다.

터미널 항목 구성은 일반 라이브러리 설치본 RSL-RL 5.4.1의 `rsl_rl/utils/logger.py`에서
iteration, throughput, collection/learning time, rolling episode statistics, elapsed/ETA 방식을 확인했다.
`policy/progress.py`의 GPU 통계 집계, 표 출력, JSONL/TensorBoard 매핑은 자체 구현이다.
모든 task 지표의 rollout 평균/최대와 dt를 반영한 보상 분해를 추가했다. 외부 logger 코드를 복사하지 않았다.

속도 개선에는 설치된 일반 Isaac Lab의 `apps/isaaclab.python.headless.kit`에 명시된
Fabric 출력 4개 OFF와 CUDA GPU interop ON이라는 실행 설정을 적용했다.
일반 Isaac Sim의 legacy `World.step(render=False)`에서도 남던 출력/readback 비용을
4096환경의 학습 없는 비교로 확인했다. PhysX 물리/충돌/관절 제약을 줄인 방법은 채택하지 않았다.
초기화 뒤 실제 GPU dynamics와 Fabric 설정을 readback해 metadata에 기록한다.
진단 코드·비교 수치·시험 결과는 Git 제외 `local/tools/profile_rollout.py`와
`local/reports/iteration_speed/`에 보관한다.
