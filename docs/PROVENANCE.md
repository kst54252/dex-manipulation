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

### 최초 정렬의 방법론 대응표

아래 표는 최초 Leap 기반 정렬의 기록이다. 현재 floating Revo2 보상·증강·관절 설정은
아래의 v4 수정 절이 이를 갱신한다.

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
- initial can +Z를 가상 table up으로 쓰는 frame은 실측 calibration이 아니다. 원 촬영 fps는 미확인이다. 초기 구현에서는 RL만 9초 horizon으로 retime했으며, 아래 v8에서 입력 시계 보존으로 정정했다. 사용자 요청으로 원본 42~50번을 제외해 현재 기하 궤적은 15~41번의 27자세, 5.2초다. 남은 자세와 0.2초 간격은 바꾸지 않는다. 잘라내기 전 원본은 `local/archive/before_tail_trim/`에 보관한다.
- REGRIND retargeter, OmniRetarget 구현, 기존 regrind-revo2의 FK/retargeting 코드는 참고·복사하지 않았다. 후속 사용자 요청으로 읽은 RL 설정/방법론은 별도 출처로 기록한다.

원본 저장소의 MIT 및 일부 모듈 BSD-3-Clause 표시를 확인했으며 원본 코드는 재배포하지 않는다. 일반 패키지는 설치본의 라이선스를 따른다. 최초 정렬의 상세 차이와 미이식 선택 기능은 로컬 대조표 `local/reports/RL_DIFFERENCES.md`에 공개한다. 최초 정렬 때는 학습하지 않았으며, 이후 사용자 요청의 학습 및 아래 진단 실험은 별도 실행이다.

### Floating Revo2 보상·물리 진단과 v4 수정 (2026-09-20)

사용자의 보상 비교 및 원인별 실제 실험 요청에 따라 로컬 `regrind-revo2`가 가리키는
`/home/wanjunkim/ARSL/regrind-upload`의 HEAD
`a52b6acfe16519fb1f26e184db82b84a1ebcaa17`을 참고했다.
아래 경로는 `regrind/source/regrind/regrind/` 기준이며 mdp/config는
`tasks/manager_based/dexterous/` 아래다. 원본 FK/retargeter 구현을 가져오지 않았다.

| 참고 부분 | 반영한 방법/설정 | 자체 구현 또는 유지한 차이 |
|---|---|---|
| `mdp/rewards.py`, `config/rb3_revo2/rb3_revo2_tuna_env_cfg.py`의 RewardsCfg, floating 파생 config | 9개 보상 항과 weight/std, dt 곱; 빠졌던 action L2 weight 0.5 복원 | `task.py`의 기존 수식을 수치 대조; 자체 workspace/coupling 종료는 유지 |
| `mdp/rb3_revo2_commands.py`, `config/revo2_floating/revo2_floating_tuna_env_cfg.py` | 에피소드 내 고정 XY 배치, 관측 위치의 증강 이동량 제거, critic의 물체 선·각속도 | 독립 `TensorReference`/`ObservationHistory`; ±5cm 범위와 사용자 지정 30cm 지지면 유지 |
| `robots/free_revo2_right_hand.py` | SI force drive 3/0.1, effort 0.5Nm, simulator velocity 100rad/s를 비교 후보로 사용 | USD degree↔SI radian 변환 및 readback 검사는 자체 수정. 실험 후 기본 gain 20/1, simulator velocity 10rad/s 선택 |
| floating 캔 config의 randomization | 캔 COM 범위 XYZ ±2/2/1mm, default joint offset 0 | 이전 screwdriver용 COM 범위 교체; 나머지 재질/질량 범위는 유지 |
| `mdp/actions.py`, `mdp/rb3_revo2_actions.py`의 floating joint/wrist 제어 부분 | 원본 follower별 PD, palm force, solver/self-collision 설정을 비교 실험 후보로 확인 | 기본 hard mimic, root force, self-collision ON, target slew, 기존 solver/dt 유지. 다른 arm IK 함수는 이 작업에 사용하지 않음 |

**Mimic 해석 정정:** 원본 floating USD의 종속 관절에도 `NewtonMimicAPI`가 있고,
설치된 Isaac Sim 6.0.1의 PhysX schema는 이를 지원한다고 명시한다. 따라서 원본을
"follower PD만 있고 hard coupling이 없는 모델"로 분류할 수 없다. 이전 adapter는
이 schema를 남긴 채 legacy PhysX mimic도 author했다. 진단에서는 한 schema만 쓰는 경우와
follower PD 유무를 별도로 비교하며, 실제 solver가 중복 제약을 만드는지는 schema 존재만으로
단정하지 않는다. checkpoint가 gain을 복원하는 비교 실험은 복원 후 실제 gain도 검증한다.
이 마지막 고정-policy 대조에서 단일 schema나 follower PD 추가는 남은 관절 속도 문제를 해결하지
못했으므로 기본 제어 방식을 바꾸지 않았다. `mimic_schema_policy: single`과
`mimic_mode: constraint_with_drives`는 명시적인 비교 옵션이며 기본 학습에 켜지지 않는다.
원본 gain 3/0.1을 적용한 고정-policy 실패를 그 설정으로 처음부터 학습해도 실패한다는 뜻으로
해석하지 않는다. 원본의 전체 학습 성능 우열을 입증한 실험이 아니다.

설치된 일반 OpenUSD/PhysX schema와 Isaac Sim API를 통해 angular drive gain 단위 및 실제 SI 값을
검산했다. SI gain을 USD에 쓸 때 `pi/180`을 곱하며 startup에서 `get_gains()`와 대조한다.
config에 단위가 없는 이전 실행은 `usd_degrees`, 증강 mode가 없는 실행은 기존 fade로 해석해
과거 checkpoint의 재현 경로를 남겼다. 새 학습 규약은 actor67/critic94이고 이전 규약과의
checkpoint hash 검사를 우회하지 않는다.

원본 함수/클래스는 프로젝트 코드에 복사하거나 런타임 의존성으로 추가하지 않았다.
무시된 로컬 테스트 `local/tests/test_policy_reward_source.py`에서만 원본의 선택된 순수 보상 함수를
AST로 읽어 실행하고 임의 상태 64개의 계산값을 비교했다. 이는 검증용 수치 기준이며 학습 구현은
`dex_manipulation`과 일반 PyTorch/RSL-RL/Isaac API만 사용한다.

검증 순서는 보상만 바꾼 512환경×256회 대조, 고정 위치 증강 대조, 128환경의 동일 random action
물리 실험, 수정 조합의 4096환경×1000회 학습이다. 진단 학습에서는 외란을 끄고
중력 일정을 10배 단축해 750회부터 9.81에 도달하도록 했다. 기본 10000회 일정은 그대로다.
이는 짧은 원인 탐색 실험이며 10000회 성능·여러 seed에서의 일반화를 입증한 것은 아니다.
모든 진단은 headless이며 원본 asset/data와 이전 학습 결과는 보존한다.
파일 hash, 실험별 설정·로그·결과는 `local/results/policy/diagnosis_20260920/`,
비교 보고서는 `local/reports/policy_diagnosis/`에 저장한다. 관절 제약과 물체 추종을 별도로 보고하며
시연 끝까지 도달한 횟수를 곧바로 파지 성공률로 사용하지 않는다.


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

## 10000 iteration 학습용 중력 일정과 비시각화 실행 (2026-09-20)

- 후속 사용자 요청으로 기본 학습 길이를 10000 iteration으로 바꾸고 위 2000회 실험의 단계 시점을
  정확히 5배 늘렸다. 초기 2000회는 중력 0, 이후 500회마다 다음 범위, 7500회부터 9.81 m/s² 고정이다.
  원래의 13개 중력 범위와 reset 시 재표본화 방식, 손의 중력 OFF, 외란 스케줄은 유지했다.
  초기 20% 무중력과 마지막 25% 정상 중력이라는 실험 시간 배분은 자체 설계이며 최적성 검증을 뜻하지 않는다.
- 사용자가 학습을 직접 요청했고 시각화를 금지했다. `--headless --skip-evaluation`으로 PPO 업데이트만 실행한다.
  설치된 일반 Isaac Sim `SimulationApp`의 `hide_ui`/`disable_viewport_updates` 설정을 이용해 UI와 viewport
  갱신을 명시적으로 끈다. Vizkit 의존이나 호출은 없으며, 종료 후 평가·재생기를 자동 실행하지 않는다.
  실행/출력 선택은 `run_metadata.json.execution`에 별도로 기록하고 policy contract hash에는 넣지 않는다.
- 속도·수치 안정성 검증은 실제 4096환경 PPO 사전 실행으로 측정하며 도구·로그는 `local/`에 보관한다.

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

## 플로팅 동작 안정화 (2026-09-20)

사용자가 허용한 로컬 `regrind-revo2/regrind/source/regrind/regrind/` 아래의
`tasks/manager_based/dexterous/mdp/actions.py`와 `robots/free_revo2_right_hand.py`를 읽었다.
비교한 범위는 매 물리 tick의 SE(3) pose PD, world-frame force, 링크 질량에 따른 torque 분배,
raw residual clipping, 관절 drive, 접촉 보정 속도와 solver 반복 수다. 외부 구현을 import하거나
클래스 본문을 복사하지 않았다. FK/retargeting 구현은 이번 작업에서 참고하지 않았다.

유지한 원본 방법은 힘/토크 기반 손목 PD, 손 중력 OFF, 캔 중력/충돌 ON, 120Hz 물리와 30Hz 정책이다.
원본의 raw-action clipping은 프레임 간 손목 속도/가속도 제한이 아니므로 빠른 방향 반전을 허용한다.

자체 설계는 `control.MotionController`의 causal 명령 생성, 이동/회전 속도 및 가속도 제한,
quaternion의 짧은 회전, 관절 제동 거리와 coupling에서 유도한 속도 제한, 환경별 reset 처리다.
물리 접촉 보정 속도는 별도 비교로 선택했다. 값은 `config/policy.json:motion_control`에 있으며
제조사 RB3 Cartesian 제한이나 논문이 제시한 값으로 주장하지 않는다. 팔에 사용할 때는 이 명령을
별도의 strict arm IK와 관절/충돌 검증에 통과시켜야 한다.

고정 정책의 원시 목표/제한 후 명령/실제 물리 상태를 따로 저장하고, 기존 checkpoint 계약은 유지한다.
재생용 제어 변경은 `execution.motion_control`과 `playback.json`에 명시한다. 접촉 수치는 PhysX
초기화 전에 적용한다. solver 반복 증가와 torque 적용 위치 변경 등 채택하지 않은 실험도
Git 제외 `local/results/floating_stability/`의 로그에 남겼다. 결과·한계는
`local/reports/floating_stability/`에 기록한다. 기하학적으로 부드러운 명령, 동적 파지 성공,
실제 로봇 제어 검증은 서로 다른 항목이다.

## 안정화 제어 1000회 재학습 (2026-09-20)

사용자의 명시적인 재학습 요청으로 4096개 환경에서 새 PPO를 1000회 학습한다.
중력 범위와 reset 적용 방식은 유지하고, 단계 시점만 기존 10000회 스케줄의 1/10로 줄인다.
200회부터 중력을 높이고 750회부터 9.81m/s²에 고정한다. 이는 이번 실행 길이에 맞춘 자체 일정이며
원본 논문의 수치라고 주장하지 않는다. RSI OFF, 증강 ON, v5 안정화 제어를 유지한다.
기존 push 시작 시점은 이번 실행 범위 밖이므로 외란은 발생하지 않는다.
기존 정책의 가중치나 optimizer를 이어 붙이지 않고 새로 초기화한다. 원본 데이터/asset은 수정하지 않는다.
GUI/뷰어를 실행하지 않고 학습과 별도 headless 평가를 수행한다. 실행 설정·소스 snapshot·checkpoint·
실패를 포함한 측정 결과는 Git 제외 `local/results/policy/stable_1000_20260920/`에 기록한다.

## 원본 접근 동작과 RSI 정합 (2026-09-20)

사용자가 로컬 `regrind-revo2` 원본 방식의 유지와 RSI 적용을 명시했다. 비교 범위는
`regrind/source/regrind/regrind/` 아래 다음 RL 구성이다.

- `tasks/manager_based/dexterous/mdp/actions.py`: reference를 기준으로 하는 clipped SE(3) residual,
  매 물리 tick의 pose PD, 지정 body에 force와 링크 질량 비례 torque.
- `tasks/manager_based/dexterous/mdp/rb3_revo2_actions.py`의 floating에서 사용하는 손가락 residual 부분:
  leader/follower 각각에 coupling target을 주고 position limits로 clip.
- `tasks/manager_based/dexterous/config/revo2_floating/revo2_floating_tuna_env_cfg.py`:
  120/30Hz, PD 수치, reset perturbation·RSI·관측 구성, solver.
- `robots/free_revo2_right_hand.py`, `objects/tuna_can.py`: force drive 3/0.1, effort 0.5,
  simulator velocity 100rad/s, 손/캔 관통 보정 5/2m/s와 물체 contact offset 2mm.
- `tasks/manager_based/dexterous/mdp/rb3_revo2_commands.py`의 RL reference/reset 부분:
  nonterminal frame 균등 정수 sampling, selected pose+velocity reset, 관절만 ±0.02rad noise,
  중앙 차분 속도(첫 frame 0, 끝 frame 후진 차분), endpoint로 정규화한 phase.
- `tasks/manager_based/dexterous/mdp/rewards.py`, `observations.py`,
  `config/rb3_revo2/rb3_revo2_tuna_env_cfg.py`의 RewardsCfg: 보상과 관측 정의 재확인.

기존 보상과 residual 범위는 원본과 일치하므로 유지한다. 원본도 접근 중 각 축 33.33mm의
위치 residual을 허용하고 손목 위치/회전 보상 가중치는 각각 0.05다. 이번 사용자 선택에 따라
파지 전 residual 강제 차단·축소나 별도 추종 보상은 추가하지 않는다.

v6는 원본처럼 목표를 직접 적용하며 추가 governor와 joint target slew 제한을 기본으로 끈다.
손목 힘 적용 body, leader/follower drive, solver, 접촉 설정을 원본에 맞춘다. 원본의
NewtonMimicAPI가 등록된 Isaac Sim 6에서는 해당 asset constraint만 보존한다. PhysX mimic을
중복 적용하지 않는다. 별도 single 모드는 Newton을 제거하고 한 개의 PhysX mimic으로 같은
관계를 표현하는 자체 호환 경로이며 실제 32환경의 zero/fixed-action 비교에서 동일한 결과를 확인했다.

`ReferenceStateSampler.uniform_frames`와 `TensorReference.control_difference`는 위 수학적
정의에 따라 일반 Torch/NumPy/SciPy로 독립 구현했다. 외부 RL 클래스를 import하거나 복사해
프로젝트에 넣지 않았다. 외부 FK/retargeting 구현은 사용하지 않는다.
재생 기본값은 checkpoint 제어로 바꿔, 학습 뒤 실행 시 제어를 묵시적으로 바꾸지 않는다.

v6 당시 유지한 자체 선택/사용자 조건: self-collision ON(원본 OFF), 사용자 지정 30cm 평면과 복합 원통,
현재 데이터 구간·9초 시간 배정·XY ±5cm 증강, 1000회용 중력 일정, 엄격한 관절/물체 추종 평가,
기존 checkpoint 계약 및 올바른 terminal observation으로 계산하는 timeout bootstrapping.
추가 governor와 옛 bin/adaptive RSI 및 옛 phase/속도 방식은 이전 결과 재현을 위해 선택 기능으로 유지한다.
외란 push의 기존 시작 시점은 현재 1000회 실행 범위 밖이다.

실제 수정 전후 측정, 원본 RL 파일 SHA-256, 회귀 테스트, 짧은 PPO 통합 검증은
Git 제외 `local/results/policy/reference_alignment_20260920/`에 저장한다. 설정 수정이 이미 학습된
v5 정책의 큰 residual을 없애지는 않으며, 새 설정의 파지 성능은 별도 재학습·평가가 필요하다.

## Floating checkpoint의 팔 결합 실행 (2026-09-20)

`policy/arm.py`는 이 저장소의 USD 기반 `ArmIK`와 명시적인 workcell 변환을 사용하는 자체
deployment adapter다. 정책을 재학습하거나 원본 REGRIND arm/FK/IK 소스를 참조·복사하지 않았다.
관측의 위치·quaternion·선속도·각속도·키포인트를 source 좌표로 역변환하고 기존 actor/history를
그대로 사용한다. 출력된 손목 target만 source→base 변환 후 flange/mount-aware IK에 전달한다.
`policy/arm_env.py`는 실제 조립 articulation의 상태를 읽고 위치 목표를 주는 Isaac 경계이며
floating root wrench를 사용하지 않는다. 이후 실제 로봇 명령으로 보내는 부분은 구현하지 않는다.

자체 선택: online IK는 이전 성공 명령의 local seed 하나, 초기화는 기존 multistart를 사용하며,
실패 후보는 적용하지 않고 episode 실패를 명시한다. 원본 target을 투영하거나 tolerance를 완화하지
않는다. 초기 구현은 손/캔의 호환되는 물성을 checkpoint 첫 환경에서 복원하고, 팔/손/책상에 nominal 마찰을 썼다.
아래 추종 오차 수정에서 손·책상 마찰 복원과 손목 응답 변환을 추가했다.
floating carrier 제거, 팔 중력과 USD drive, 큰 책상, 팔 도달 범위는 학습과 다른 배치 조건이다.
고정 base/self-collision/실제 joint limit·velocity·singularity 지표를 검사하지만 충돌 회피 planner나
물리적 파지 성공 인증은 아니다. Isaac 미실행 CPU 검증은 Git 제외 `local/results/policy/arm_adapter_20260920/`에 남긴다.

후속 Isaac 런타임 검사에서 손의 `NewtonArticulationRootAPI`가 implicit
`PhysicsArticulationRootAPI`를 포함해 root가 두 개로 구성되는 차이를 확인했다.
`sim.prepare_arm_articulation`은 실제 base의 root를 선택하고, 활성 joint graph와 world 고정 연결을
검증한 후 연결된 손의 중복 root schema만 실행 stage에서 제거하는 자체 호환 처리다.
독립적인 다른 articulation은 병합하지 않으며 원본 asset과 checkpoint를 수정하지 않는다.
runtime schema 조사·회귀 테스트·실제 headless 정책 재생은 Git 제외
`local/results/policy/arm_runtime_fix_20260920/`에 기록한다. 파지 실패도 그대로 남긴다.

## 원본 성공 실행 및 재생 시간 계약 재검증 (2026-09-20, v7)

비교 범위는 local original의 RL 환경/로봇/물체/commands/actions/rewards/observations 설정,
`logs/rsl_rl/floating_revo2_tuna/2026-09-05_16-46-54_floating_stable_ground_5000/params/{env,agent}.yaml`,
5000/10000회 TensorBoard 기록과 입력 HDF5의 데이터입니다. 원본 FK/IK/retargeting 소스는 읽거나 재사용하지 않았습니다.
기존 9초 resampling을 'source-style'이라고 표현한 것은 이 원본 Revo2 command에는 맞지 않았습니다.
원본은 각 입력 프레임을 한 control tick에 소비하고 episode timeout과 별개로 demo 끝에 종료합니다.
첫 대조 실험(v7)에서는 27개 자세를 27 control tick으로 재생했습니다. 이는 별도로 선택한 진단용 재생 속도이며 capture FPS 주장은 하지 않습니다.

원본에서 참고하여 맞춘 항목: 외력 매 TGS iteration 적용, GPU partition 수, self-collision OFF,
body-name에 따른 물성 randomization 대상/범위, critic의 fingertip link-origin 관측입니다.
보상 수식/가중치와 PPO 설정은 기존과 동일하며 별도 로컬 수치 oracle 테스트로 확인합니다.
기존 collider와 원본 collider의 상대 거리 비교는 본 프로젝트의 FK/FCL로 수행했습니다.
현재 reference에는 thumb/can 간격이 남아 있고 원본은 thumb 접촉에 더 가까웠습니다.
이는 입력 기하 차이의 진단이며 원본 궤적을 새 reference로 복제한 것이 아닙니다.

자체 설계: 전체 horizon 평가에서 실패 프레임을 제외하지 않는 평균/최대 오차 및 엄격한 완료 지표,
과거 checkpoint의 타이밍을 보존하는 명시적 legacy 모드, 1000회 안에서 정상 중력에 도달하는 일정.
사용자 형상(73mm 몸통, 76mm×3mm 하단), 30cm 지지면, 15~41번 자세와 XY±5cm 증강은 유지합니다.
게인 20/1, 속도 제한 변경 등의 대조 실험 결과는 local에 보존하고 최종 v7 설정에는 가져오지 않았습니다.
원본 asset/data는 수정하지 않으며 실험, snapshot, checkpoint와 상세 분석은 Git 제외 local에만 저장합니다.

원본의 자동 reset 순서도 반영합니다. RSI로 실제 상태 k를 복원한 뒤 command manager가 k+1로 진행하므로, 자동 reset 직후의 다음 목표는 k+1입니다. 외부에서 호출한 초기 reset은 k를 유지합니다. 32개 실제 PhysX 환경에서 위치 복원과 다음 명령 프레임을 따로 검증했습니다. 책상 torsional patch의 자체 고정 반경(20/5mm)도 원본 기본값(0/0)으로 되돌렸습니다.

추가로 성공 실행 reference의 HDF5 계보를 확인했습니다.
`rb3_revo2_reference_stable.h5`의 wrist/object pose와 손 관절은
`outputs/floating/random_can_replay/20200709_143747_left_random_rollout.h5`의 실제
재생 상태에 기록된 alignment와 frame 선택을 적용한 값과 정확히 일치합니다.
그 재생의 원래 object target과 실제 object position은 평균 13.167mm 차이납니다.
따라서 원본 학습 그래프의 약 1mm는 이미 물리 재생 결과를 reference로 삼은 오차이며,
본 프로젝트의 원래 object trajectory에 대한 오차와 직접 동일시할 수 없습니다.
데이터 배열과 메타데이터만 조사했으며 원본 retargeting/FK/IK 코드를 사용하지 않았습니다.
본 프로젝트의 진단 후보도 원래 object trajectory를 실제 결과로 바꾸지 않습니다.
실험 입력·검증 기준의 변경은 별도 파일과 metadata에 기록합니다.

## 자체 물리 접촉 reference 준비와 입력 시계 보존 (v8)

원본 성공 실행에서 참고한 방법은 물리 rollout의 손-물체 접촉 자세를 reference의 출발점으로
사용하는 것입니다. 원본 데이터·정책 가중치·retargeting 코드를 가져오지 않았습니다.
`policy/prepare.py`는 본 프로젝트 v7의 평가 rollout, 사전에 고정한 환경 index 0을 사용합니다.
`T_hand_new = T_object_target @ inverse(T_object_measured) @ T_hand_measured`로 접촉 관계를
원래 물체 목표에 맞추고, 측정 독립 관절을 기존 모델의 coupling으로 확장합니다.
일반 SciPy SLSQP와 본 프로젝트 FK/FCL을 사용한 작은 collision projection, 전체 collider의
손-물체/손-바닥 검사 및 보간 substep 검증은 자체 설계입니다. 목적 물체 pose를 실제 결과로
교체하여 오차를 줄이는 변경은 하지 않습니다. 원본 input은 보존하고 별도 local 파일을 생성합니다.

v8은 `input_timestamps`로 입력의 5.2초를 그대로 유지합니다. 30Hz control grid의 157개
명령 시점에 보간하며, 10초 timeout과 분리합니다. 이 시간은 사용자가 허용한 느린 재생 시간이고
측정한 capture FPS가 아닙니다. 27개 자세를 원본의 30Hz FPS로 간주하지 않습니다.
보상, PPO, PD, RSI, 물성 범위는 v7의 원본 정합 설정을 유지합니다. 접촉 reference의 실제
파지 성능은 별도 새 학습 및 정상 중력에서 전체 시퀀스 평가로 확인합니다.

유지한 자체 정확성 개선: 현재 물체 선속도는 COM 속도에서 `omega × COM_offset_world`를
빼 pose 원점 속도로 변환하고 reset은 역변환합니다. 원본 RL은 object reference의 원점 차분과
Isaac Lab `root_lin_vel_w`의 COM 속도를 혼용합니다. 입력 object pose 기준과 일치하는 현재
변환을 유지하며, 원본 손목 PD의 `root_link_vel_w`와는 같은 기준점을 사용합니다.

## 팔 연결 정책의 추종 오차와 물리 응답 수정 (2026-09-20)

비교 대상은 이 저장소의 동일한 v8 checkpoint, 동일한 물체 목표, 실제 Isaac Sim 손목/관절 상태다.
원본 REGRIND/Revo2 FK·IK·retargeting 소스를 읽거나 복사하지 않았다. 일반 Isaac Sim의 설치된
`Articulation`, `RigidPrim`/contact view API 구현을 확인해 관절 속도 목표, shape material 및
접촉 impulse/dt 단위를 확인했다. 원본 방법에서 유지한 부분은 기존 floating 정책의 residual 출력과
pose PD 의미·게인, 물성 randomization 결과, 관측 및 actor다. 이번에 원본 arm controller를 가져온 것은 아니다.

자체 수정은 다음과 같다.

- 팔 위치 drive의 속도 목표가 0으로 남던 경로를 수정해, strict IK를 통과한 목표 차분 속도를 전달한다.
  native USD gain·effort는 변경하지 않고 IK 실패는 이전 목표/속도 0과 명시적 실패로 처리한다.
- 정책 PD 목표를 즉시 실현 가능한 wrist pose로 취급하지 않는다. `policy/arm_control.py`의 순수
  NumPy/SciPy 합성 강체 모델이 학습 질량·관성·COM·PD로 120Hz 응답을 계산하고 strict IK에 전달한다.
  composite COM의 parallel-axis inertia, palm COM에 작용하는 PD 힘의 moment와 gyroscopic 항을 포함한다.
  floating carrier frame/COM은 자체 USD에서 읽으며 실제 조립 팔에 carrier 질량을 추가하지 않는다.
- 실제 hand link 접촉력 합과 body COM 기준 근사 모멘트를 응답 모델에 되먹임한다. 필터 alpha 0.5,
  피드백 gain 0.25는 지역 진단에서 선택한 자체 배치 설정이다. gain 1/0.5와 free-space-only 모델의
  실패 결과도 local에 남긴다. 정확한 접촉점 torque와 손가락 내부 운동량을 재현하는 모델은 아니다.
- 빠진 손 material 복원을 이름/shape 수 검증으로 추가하고, workcell 크기·배치를 유지하며 책상의
  고정 kinematic contact/solver/material 및 scene switches를 floating 학습 설정과 맞춘다.
- raw 정책 목표→측정 손목, 응답 모델 목표→팔 FK, 팔 FK→측정 손목 오차를 분리해 저장한다.
  floating 학습의 시간 계약은 유지하며 팔 재생의 endpoint 중복만 정수 frame count로 제거한다.

원래 asset/data·checkpoint·물체 목표는 수정하지 않았으며 RL 재학습을 하지 않았다.
실제 하드웨어의 force sensing/driver·calibration은 구현하지 않았다. 순수 관절 목표 출력을
실물에서 안전하게 재현할 수 있다는 인증도 아니다. headless 실측·실패와 제외한 실험은
Git 제외 `local/results/policy/arm_tracking_20260920/`에 보존한다.

## 원본 마운트 부품을 이용한 일자 조립 적용 (2026-09-20)

사용자 요청으로 `regrind-revo2/USD/rb3_revo2_vertical.usda`와
`USD/revo2_vertical_adapter/revo2_vertical_adapter.usd`의 asset을 확인했다.
두 파일은 이미 이 저장소에 보관한 동명 asset과 SHA-256이 일치했다. 원본 마운트의 visual/collision,
link6→adapter 및 link6→hand attachment를 그대로 사용한다. 새 외부 코드 복사나 원본 FK/IK 소스
참조는 하지 않았다. 기존 꺾인 USD와 asset/data는 수정하지 않는다.

자체 작업은 기본 IK/재생/정책 설정을 일자 조립으로 전환하고, 현재 USD extractor로 별도
`assets/models/rb3_vertical.json`을 생성한 것이다. 손 +Z와 link6 +Z는 일치하며 wrist 원점은
link6에서 약 141.305mm 떨어진다. 원래 reference·물체 pose·책상 배치 회전은 보존하고 새 장착에 맞게
초기 IK seed와 팔 joint trajectory를 다시 계산한다. 기존 궤적은 Git 제외 local에 보관한다.
재생기는 mount fingerprint가 다른 오래된 궤적을 거부하고, 정책과 동일한 articulation root
검증 함수를 사용한다. 상대 USD 참조, adapter collision, 독립 FK/IK와 Isaac 실제 link pose를 검증하며
결과는 `local/results/straight_mount_20260920/`에 기록한다. 이전 꺾인 장착의 파지 성능을 새 장착의
검증 결과로 재사용하지 않는다.

## 원본 사람 데모의 독립 리타게팅과 별도 학습 (2026-09-20)

사용자 요청으로 원본 프로젝트의 launcher 설정과 데이터 metadata에서 기본 시퀀스
`20200709_143747_left`를 확인했다. 입력은
`outputs/preprocessed/dexycb/20200709_143747_left/dexycb_right_hand_preprocessed.npz`의
사람 MANO21 3D annotation과 object pose이며 원본 ID 12~51의 40프레임이다.
원본 `docs/DATA_PIPELINE.md`에 기록된 camera −Y-up 규약과 metadata의 재생 시계 30Hz를
참고했다. 원본 FK/IK/retargeting 구현은 읽거나 복사하지 않았고, 물리 rollout에서 만들어진
원본 stable robot reference와 기존 정책 가중치도 새 학습의 입력으로 사용하지 않았다.

재사용한 데이터는 원본 왼손 annotation을 object-local X 반사하여 오른손으로 변환한
`mano_joint_coords_right_mano21`와 반사하지 않은 object pose다. 자체 adapter는 원본 왼손
annotation으로 이 변환을 다시 계산해 일치 여부를 검사하고, quaternion 순서와 sequential
MANO21 의미를 검증한다. 원본 raw camera label의 사람 좌표 및 grasped object ID 6 pose와도
수치 대조했다. 소스 SHA-256과 변환 오차를 output metadata에 남긴다.

자체 구현/선택은 `scripts/demo.py`와 `data.import_human_demo`, 명시적 semantic correspondence,
camera −Y-up grounding 선택, 입력 frame ID 기반 timestamp 보존, 원본 파일 덮어쓰기 방지다.
현재 프로젝트의 FK·interaction-mesh 최적화·collision 검사를 그대로 실행해 새 손목/손 관절
reference를 생성한다. 캔은 사용자 지정 Ø73×32mm 및 하단 Ø76×3mm 형상과 기존 50점을 유지한다.
카메라 −Y-up은 원본의 좌표 규약이며 실측한 책상 calibration이나 capture FPS를 주장하지 않는다.

원본 데모의 initial object +Z를 지면 위쪽으로 사용하면 물체가 들릴수록 지면 아래로 내려가는
차이가 있어 이 데모에만 grounding mode를 명시했다. 손과 물체에 같은 SE(3)를 적용하고,
첫 프레임 collision 밑면을 Z=0에 놓는다. 프레임별 pose나 목표 물체 궤적을 물리 결과로 바꾸지
않는다. 기존 기본 데이터와 과거 checkpoint의 지면/시간 규약은 유지한다.

새 실험은 기존 v8의 원본 정합 RL 설정에서 별도 입력 reference로 4096 환경·1000 iteration을
처음부터 학습한다. RSI와 증강을 켜고 정상 중력 도달 시점을 750 iteration으로 유지한다.
전체 horizon 평가는 RSI·증강 없이 frame 0부터 정상 중력에서 실시하며 실패 뒤의 오차도 포함한다.
실험 데이터·테스트·학습 snapshot·checkpoint·결과는 Git 제외
`local/results/original_demo_20260920/`에 별도로 저장한다.

## 사용자 지정 팔 환경의 캔 거리 55cm (2026-09-20)

사용자 요청으로 기본 팔 배치의 초기 캔 XY를 (0.30, 0)에서 (0.55, 0)m로 옮겼다.
거리는 로봇 베이스 중심과 캔 pose 원점 사이의 수평 거리다. 같은 고정 SE(3)를 손·물체
전체 궤적에 적용하므로 상대 자세·높이·시계·정책은 바꾸지 않는다. 기존 기본 27프레임
데모는 −90도 방향을 유지하고 IK 산출물을 다시 생성했다. 이전 산출물은 local에 보관한다.

새 원본 휴먼데모의 camera −Y-up frame에 기존 −90도 회전을 적용하면 손끝이 로봇을 향하고
55cm 위치에서 IK가 도달하지 못했다. 이전 사용자 요청인 손끝 외향 배치를 위해 이 데모의
별도 arm config에서는 +90도를 사용한다. 이는 로봇 마운트 변경이 아닌 손·캔 전체의 동일한
world yaw이며 원본 데이터·checkpoint를 수정하지 않는다. 원본 FK/IK/retargeting 구현은
참조하지 않았다. raw reference의 속도 제약 실패와 물리 재생 결과를
`local/results/original_demo_20260920/arm_55cm/`에 함께 기록하며 완전한 팔 추종 성공을 주장하지 않는다.

## 원본 휴먼데모의 뒤집힌 캔 정렬 수정 (2026-09-20)

사용자 지적으로 실제 캔 collision geometry와 학습에 저장한 reference를 대조했다.
초기 원본 휴먼데모 실험은 40/40프레임에서 Ø76×3mm base가 body보다 위에 있었다.
camera −Y-up을 올바르게 적용했지만 raw object pose에 대체 비대칭 형상을 identity로 붙인 것이
원인이다. 해당 1000회 checkpoint와 약 6.7mm 평가는 **뒤집힌 캔 조건**의 결과이며 정상 방향의
학습 결과로 인정하지 않는다. 이전 기본 27프레임 reference의 초기 방향은 정상이다.

자체 수정: `geometry.can_base_down_alignment`가 실제 Base/Body collider와 명시된 기하 중심을
읽어 한 번의 고정 asset-to-pose SE(3)를 계산한다. 이 데모의 보정은 local X 180도 회전과
`center - R @ center` translation이다. 중심 오프셋이 있는 USD이므로 pose 원점 주위의 단순
회전으로 대신하지 않는다. raw 물체 annotation·오른손 annotation·시간은 그대로 보존하고,
같은 50개 local surface point 및 USD 전체 물성/형상을 새 object pose에 함께 적용한다.
월드나 사람 동작 전체를 뒤집지 않는다. 원본 FK/IK/retargeting 코드는 사용하지 않았다.

새 alignment로 독립 리타게팅과 grounding을 다시 생성하며, grounding·새 학습 진입 전에
초기 base/body 높이 관계를 검사한다. 과거 checkpoint 검사/재생의 hash 계약은 보존하지만,
실제 새 학습은 upside-down reference로 시작할 수 없게 한다. 테스트와 실제 PhysX 초기 방향
검증은 local에 저장한다. 수정된 입력에 대한 새 정책 학습은 아직 수행하지 않았으며 기존
정책의 성능 수치를 재사용하지 않는다.
## User-authorized second-demo timing (2026-09-21)

The user requested the imported second motion to run slowly like the first.
The existing first reference lasts 5.2 seconds; the corrected base-down second
reference lasts 1.3 seconds. A derived copy uniformly multiplies its timestamps
by four, preserving all 40 poses, joint coordinates, keypoints and annotations.
The second experiment's policy and arm configuration now point to this same
derived input. Playback timing remains explicit, independent of capture FPS;
30 Hz control and 120 Hz physics settings are unchanged. This is a local timing
choice, not a method taken from REGRIND/OmniRetarget. Original inputs and previous
configuration snapshots are retained. No learning or physical grasp evaluation
was performed as part of this timing change.

## Frozen policy trajectory tracking (2026-09-21)

At the user's request, `policy/tracking.py` and `scripts/tracking.py` separate
policy generation from physical arm execution. One strict floating rollout is
recorded; its raw wrist PD setpoints and named finger targets are frozen. The
existing independent USD FK/strict IK creates a saved transport-neutral joint
reference. Isaac replays only these saved position/velocity targets; no actor,
online IK, virtual wrist response or contact-dependent command update runs in
the playback loop. Checkpoint loading restores physical properties only.

This diagnostic protocol, initial reset interval accounting, integer uniform
time scaling, endpoint/all-physics-tick error reporting, and final hold test are
local engineering choices, not methods copied from REGRIND or OmniRetarget.
Measured floating motion is an explicitly separate input option and is never
silently substituted for policy targets. Failed joint limits/IK are retained
as diagnostics and rejected by the replay loader. Physics uses the captured
checkpoint condition, including historical object orientation; it does not
claim a new policy for a corrected or retimed dataset. Tests and measured runs
remain under ignored `local/`.

## 데모 파일 정리 (2026-09-21)

- 자체 구조 선택: `data/current/`, `data/original/` 두 데모 폴더에 원본 annotation, camera 리타게팅, 현재 ground reference, 기존 정책 학습 reference를 역할별로 배치했다. 알고리즘·물리·시간·정책 가중치는 변경하지 않았다.
- 외부에서 가져온 파일은 원본 프로젝트의 `20200709_143747_left` 전처리 사람 데모 NPZ다. `regrind-revo2`는 이 머신에서 `regrind-upload`를 가리키며, 복사 SHA-256은 `b65141ff4bd34deeca1a10024fe3cb44e0798a0c83ab55d0efb5f13886929158`이다. 원본은 수정하지 않았고 외부 구현이나 정책 가중치는 가져오지 않았다.
- 최신 upright·5.2초 기하 reference와 수정 전 1.3초 정책 학습 reference를 구분한다. 기존 NPZ/변환 기록과 checkpoint는 바이트를 보존했고, 기존 `local/` 입력 경로에는 상대 링크를 둔다. 이전 data 경로를 포함한 출처 기록은 `resolve_demo_path`로 해석한다. [데모 구조](data.md).

## 상판 충돌 보상과 명령 보호 (2026-09-21)

- 실제 120Hz PhysX 재생에서 두 기존 floating 정책의 손–상판 접촉을 확인했다. 상판만 필터링한 접촉 sensor로 캔 접촉을 분리했고, collision mesh 전체 vertex의 최저 높이도 함께 기록했다. 진단 코드·결과는 무시되는 `local/tools/check_table_contact.py`, `local/reports/table_contact_20260921/`에 둔다.
- 자체 설계: collision mesh의 링크별 local AABB를 rigid pose로 변환한 보수적 최저 높이, 동일 USD joint frame/axis/coupling/reversed 연결의 Torch FK, 실제 physics substep 최소 여유와 **보호 전** 목표 여유의 제곱 비용, 손목 Z 명령만 올리는 공통 보호 처리를 추가했다. 4mm 여유·20ms 속도 선행·비용 8/4·clamp 2/3은 이번 프로젝트의 선택이며 REGRIND/OmniRetarget에서 가져온 수식이나 설정이 아니다. 12mm/50ms 초기 후보는 기존 물체 추종을 과도하게 바꿔 채택하지 않았다.
- 보수적 상자는 semantic keypoint와 다르게 모든 추출 collision vertex를 포함한다. 무한한 상판 반공간을 사용하므로 테이블 밖에서도 아래로 내리지 않는다. 속도 선행은 선형 예측 보조이며, 이 방법을 연속 시간 충돌 회피 증명·일반 self-collision 검사·실물 안전 제어로 보고하지 않는다.
- 플로팅과 팔에 동일한 policy-source frame 검사를 적용한다. 팔에서는 virtual wrist response 출력도 검사하고 strict IK의 도달성·singularity·step/velocity 제약은 유지한다. 팔 링크/받침대 충돌 전반을 이 손 보호로 검증했다고 주장하지 않는다.
- 기존 PPO/물체 추종 보상은 유지하고 안전 비용만 추가했다. 저장된 정책·입력·asset은 수정하지 않았으며 새 설정은 새 학습 계약을 만든다. 기존 정책 재생의 보호 override는 checkpoint 계약 외 실행 metadata에 명시한다. 외부 프로젝트 소스 코드를 복사하거나 새로운 외부 알고리즘을 가져오지 않았다.


## 실제 팔을 포함한 residual RL (2026-09-21)

- 사용자의 제안에 따라 같은 12개 wrist/finger residual action을 물리 RB3+Revo2 환경에서 학습하도록 확장했다. 이번 확장에서는 외부 REGRIND/OmniRetarget 소스를 새로 읽거나 복사하지 않았다. 기존 독립 구현의 reference/관측/물체 보상/RSI/augmentation과 일반 RSL-RL PPO를 재사용한다.
- 자체 설계: USD 추출 arm model에 기반한 Torch batch FK/IK, 매 제어 tick의 IK 제약과 실제 articulation 동역학을 포함하는 PPO 환경, 추가 20개 arm 관측, IK·실제 추종·실패·특이점·joint-limit 비용, 실패 시 arm/finger 목표 동시 유지와 4회 연속 실패 종료다. REGRIND 논문의 arm-learning 방법이나 원본의 검증된 recipe라고 주장하지 않는다.
- FK/IK는 기존 자체 CPU solver와 같은 axis/양 joint frame/mount/limits 및 최종 FK 허용치를 사용한다. batch 계산은 고정 12회 adaptive damped least squares, float32 수치 damping 하한, cost가 감소하는 full/quarter trial, 직전 해 기반 국소 continuation을 사용한다. CPU solver의 multistart/수렴 iteration 수를 그대로 복제하지 않는다. 초기 RSI는 40회이며 구간 특이점 검사 샘플은 21개다. 수치 실패를 reachable 판정으로 바꾸지 않는다.
- 실제 팔에는 원래 USD drive gain/effort를 유지하며 hand mass/material/gain만 기존 범위에서 randomization한다. 기존 floating carrier, root wrench, 가상 손목 PD 응답은 새 arm 학습 경로에 들어가지 않는다. arm과 물체는 world gravity curriculum, hand는 gravity OFF다. fixed-base root velocity push는 하지 않는다. 실제 손/물체 접촉과 arm 추종에 의해 보상이 정해지며 rollout 중 object pose는 바꾸지 않는다.
- actor 67→87, critic 94→114 확장은 별도 checkpoint 계약이다. 선택적 floating actor warm start는 같은 입력/관절모델/residual scale/상판 설정만 허용하며 새 입력 layer 가중치를 0으로 둔다. 평균 actor와 기존 normalization prior만 이관하고 critic, optimizer, 탐색 std는 새로 만든다. prior count를 초기화하면 이미 학습된 67개 입력의 의미가 첫 rollout에서 급변할 수 있으므로 원래 count를 유지한다. 초기 학습률 0.0001 및 std 0.15는 팔 적응을 위한 자체 설정이다.
- 기존 상판 geometry guard는 유지하고 raw 명령 비용도 남긴다. self-collision OFF 설정 역시 유지하므로 이 결과는 arm-hand 무충돌 인증 또는 실물 제어 결과가 아니다. 추가 방법·명령은 `docs/arm_policy.md`, 테스트와 GPU 실행 기록은 ignored `local/reports/arm_training_20260921/`에 둔다.


## 데모 번호 (2026-09-21)

이 항목과 다음 두 항목은 당시 번호를 기록한 이력이다. 현재 번호는 아래 **데모 번호 교환**을 따른다.

사용자 요청으로 `data/current/`의 27자세 동작을 **1번**, `data/original/`의 40자세 동작을 **2번**으로 부른다. 두 데모를 계속 함께 사용한다. 실행 메뉴·CLI·등록 정책의 표시 이름을 번호로 바꾸고 기존 실행 이름은 별칭으로 유지했다. 입력·학습 설정·정책 가중치·checkpoint 계약 경로는 변경하지 않았다. `2-arm-trial`은 2번 데모의 팔 학습 시험 정책을 뜻하며 세 번째 데모가 아니다.


## 데이터 폴더 번호 적용 (2026-09-21)

사용자가 실제 폴더명도 번호로 바꾸도록 요청하여 `data/current/` → `data/demo1/`, `data/original/` → `data/demo2/`로 이동했다. 앞선 데모 번호 항목에서 유지했던 디렉터리 이름을 이번 요청으로 변경했다. `data/`에는 두 폴더만 두며, 중복 사본이나 옛 이름의 링크를 추가하지 않았다. 원본 annotation·RGB·NPZ·좌표/시간 출처 기록의 바이트는 보존하고 manifest의 데모 ID/표시 이름만 갱신했다.

자체 호환 처리: `resolve_demo_path`가 저장된 설정과 출처의 옛 경로를 새 입력으로 연결한다. 기존 `local/` 입력 링크 10개는 새 위치를 가리키게 했다. 학습 계약 해시를 계산할 때에는 정확히 두 데이터 폴더 경로의 표기만 옛 이름으로 정규화한다. 저장된 checkpoint·설정 스냅샷은 다시 쓰지 않았으며, PPO의 엄격한 해시 비교 및 reference SHA-256·모델·물리/제어/보상 검증은 유지한다. 데이터 폴더 변경 외 설정 차이를 허용하는 우회는 추가하지 않았다.

검증: 명명 manifest 외 70개 파일의 바이트 불변과 두 manifest의 파일 해시 확인, 1·2번 floating/arm 및 retarget/policy 경로 확인, 실제 저장된 정책 4개를 CPU에서 로딩하고 물리 설정 변경 시 거절되는지 확인, 기존 단위 테스트 147개 통과. 기록은 `local/reports/demo_directory_rename/`에 둔다. 이 변경으로 학습이나 GUI 재생을 시작하지 않았다.

## 학습 실행기 (2026-09-21)

자체 사용성 개선: `train.sh` / `run.sh train`이 기존 `scripts/policy.py`를 호출한다. 데모 번호, 물리 환경,
학습 횟수, 저장 위치와 이어 학습을 연결하며 새 알고리즘이나 외부 프로젝트 코드를 가져오지 않았다.
기존 설정 파일과 데이터는 유지하고 실행별 설정 사본을 ignored `local/`에 기록한다. 1번 팔 학습은
기존 팔 recipe에 1번 학습 reference·좌표 설명·팔 배치만 선택한다. 새 학습의 중력 단계 시점은 기존
recipe의 시간 비율로 재계산한다 (현재 전체 중력은 75% 지점). 중력 값 자체는 유지하며 이어 학습은
checkpoint의 원래 설정·스케줄을 그대로 복원한다. 기본은 headless 학습 전용이며 종료 후 평가나
시뮬레이터 창을 열지 않는다. 방법론·보상·물리·checkpoint 계약 검사는 기존 구현을 사용한다.

## 데모 번호 교환 (2026-09-21)

사용자 요청으로 기존 demo1과 demo2를 교환했다. 현재 **demo1은 가져온 40자세 동작**
(`20200709_143747_left`, frame 12~51), **demo2는 기존 27자세 동작** (frame 15~41)이다.
`data/`의 두 실제 디렉터리, manifest·README, 학습·재생 메뉴, 설정/스크립트의 입력 경로,
사용 문서와 `local/` 입력 상대 링크 10개를 같은 동작에 맞춰 갱신했다. 임시 세 번째 이름을 거쳐
폴더를 교환해 덮어쓰기를 피했으며 원본 NPZ·RGB·annotation·좌표/시간 기록은 바이트를 보존했다.

자체 호환 처리: `data/original`은 현재 `data/demo1`, `data/current`는 현재 `data/demo2`로 연결한다.
checkpoint 해시의 안정적인 경로 표기도 이 데이터 정체성을 따른다. 기존 정책 가중치·저장된 학습
설정·과거 실험 기록은 수정하지 않았으며, 번호가 같다는 이유로 다른 궤적을 허용하는 예외는 추가하지 않았다.
현재 저장된 정책들은 과거 original/current 또는 기존 local 링크를 사용하므로 새 위치의 동일 입력을 읽는다.
이전 검증 보고서의 번호는 작성 시점의 번호이며 재작성하지 않는다. 실제 입력 SHA-256과 엄격한
모델·물리·관측 계약 검사로 호환성을 검증한다. 이번 변경에는 재학습이나 GUI 실행이 포함되지 않는다.

## 데모1 기본 정책의 캔 방향 수정 (2026-09-21)

사용자가 정책 실행에서 캔의 두꺼운 부분이 아래로 향하도록 요청했다. 원인은 `policy 1`이 방향 수정 전
1.3초/1000iter checkpoint를 가리키던 실행 목록이었다. 이를 이미 밑면 아래·5.2초 입력으로 학습 완료한
`original_slow_table_safe_2000_20260921_035416/policy.pt`로 연결하고, 팔 배치는 같은 reference의
`config/ik_original.json`으로 선택했다. 이것은 실행 목록 수정이며 외부 알고리즘이나 코드 도입이 아니다.
물체 pose/mesh만 뒤집어 checkpoint의 학습 계약을 바꾸지 않고, 해당 정책의 저장된 설정과 원래 reference를
읽는다. 과거 정책·입력 bytes는 보존했다. 최신 정책은 floating 학습 결과이며 팔 연결 재생을 팔 학습
성공으로 보고하지 않는다. 이번 변경으로 학습이나 GUI를 실행하지 않았다.

## 데모1의 첫 캔 밑면 수평화 (2026-09-21)

사용자가 시작 시 캔이 움직이는 원인을 확인하고 바닥에 안정적으로 놓이도록 요청했다. 기존
camera -Y-up 바닥 변환은 최저 충돌점만 Z=0으로 옮겼다. 첫 캔 축이 수직에서 11.2708105°
기울어져 하단 Ø76mm 밑면의 높이 차가 14.85394mm였고, 초기 속도는 이미 0이었다. 밑면이
아래를 향한다는 검증만으로 밑면 전체가 바닥에 수평 접촉한다는 뜻은 아니었다.

자체 데이터 보정: 캔의 첫 +Z축을 world +Z로 맞추는 최소 회전을 손·캔·사람 점·로봇 점·다른 annotation
물체에 공통 적용하고, 최초 밑면을 Z=0에 맞췄다. 원점 높이는 27.15845mm에서 20.11950mm가 됐다.
기록된 후속 자세에는 새 바닥 아래로 들어가는 구간이 있어, 그 프레임의 손·물체·점 전체에 같은 Z
이동(최대 2.49482mm)을 적용했다. 이 두 번째 처리는 고정 좌표 변환과 별도의 파생 reference 보정이며
`floor_stabilization.post_frame_translation_m`에 명시한다. 손–캔 상대 자세, 관절, local geometry/점 번호,
40개 frame ID와 5.2초 시각은 유지한다. 원래 camera 좌표로 돌아가는 viewer도 이 추가 이동을 제거한다.
논문에서 가져온 기법이 아니라 사용자가 지정한 시뮬레이션 초기 조건을 위한 자체 설계다.

`data/demo1/stable/`에 새 입력을 만들고 기존 raw/camera/grounded 입력·정책 가중치를 보존했다.
새 학습/리타게팅은 stable 입력과 `config/ik_demo1.json`을 사용한다. 기존 2000iter와 팔 20iter
정책은 과거 reference 및 `config/ik_original.json`을 유지하고 메뉴에 수평 교정 전임을 표시한다.
reference SHA-256과 strict checkpoint 검사는 유지하므로 새 입력에는 새 학습이 필요하다.
첫 프레임은 기존 reset 코드의 선속도/각속도 0을 사용한다. RSI의 중간 프레임 상태 초기화, reward,
PD/물성, 중력 스케줄, 물체 자유 강체 동역학은 바꾸지 않았다. 실행 중 물체를 고정하거나 매번 pose를
덮어쓰는 보정은 없다. 이 바닥은 실측 camera–table 보정이 아니라 초기 캔 기준의 시뮬레이션 가정이다.

검증: 120Hz로 보간한 625개 자세의 최대 바닥 침범은 수치상 0.000184mm 미만이다. strict 팔 IK는
79/79 성공, 최소 정규화 singular value 0.246938이고 관측한 특이점 근접 프레임은 없다. GPU PhysX
9.81m/s²·120Hz에서 손을 떨어뜨린 두 환경의 2초 정지 시험을 했다. 기존 캔은 최대 중심 낙하 7.0393mm,
수평 이동 5.1244mm, 새 캔은 중심 낙하 0.000369mm·수평 이동 0.000537mm였다. 새 자세의 최종 밑면
침범은 약 0.00106mm였다. 이 측정은 수동 정지 안정성 검증이며 정책 추종/파지 성공은 아니다.
GUI와 정책 학습은 실행하지 않았다. 기록은 ignored `local/reports/demo1_flat_start/`에 둔다.

## Localized Revo2 fingertip contact (2026-09-21)

사용자 요청으로 별도 collision mesh가 있는 다섯 `right_*_touch_link`에만 고무 근사 접촉을
추가했다. 이 대응은 현재 Revo2 USD와 BrainCo description의 원본 URDF의 visual/collision mesh
구분을 확인한 결과다. 손끝 landmark인 `tip_link`나 손가락 외피 `distal_link`는 패드로 간주하지 않았다.
원본 asset·입력·질량·관절·gain은 수정하지 않고 실행 stage의 physics material binding을 사용한다.

접촉법은 [NVIDIA Omni Physics compliant contact 문서](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-materials-for-compliant-contacts)의
암시적 force spring/damper를 적용했다. REGRIND/OmniRetarget의 물성값이나 코드를 가져온 변경이 아니다.
정적/동적 마찰 1.2/1.0, 강성 20,000 N/m, 감쇠 20 N·s/m, average 결합은 자체 설계한 **미보정 초기값**이며
실제 Revo2 고무의 측정값으로 주장하지 않는다. 기존 rigid hand/can/table의 물성은 유지한다.
캔–패드 접촉은 pad compliance를 사용하지만 캔 자체와 캔–책상 접촉은 rigid로 유지된다.

현재 PhysX tensor API에서 getter의 compliance 값은 (강성,감쇠), setter는
(정적마찰,동적마찰,강성,감쇠)와 별도의 세 combine mode를 받는다. 강체 readback의 강성은 +infinity다.
이를 강체에 다시 compliant setter로 쓰지 않고 pad-only rigid view로만 갱신한다. 기존 startup randomization,
저장 물성 복원, 팔의 body-name material transfer 뒤에 pad override를 적용하고 실제 readback을 검사한다.
기존 checkpoint 계약을 유지하며 재생의 물성 차이를 별도 실행 metadata에 기록한다.
`--contact-materials checkpoint`는 학습 당시 물성을 복원하고, 신규 학습은 전체 프로필을 설정에 저장한다.

Isaac Sim 6.0 환경의 플로팅 2개 환경 및 RB3+Revo2 환경에서 패드 5개만 compliant인 것을 확인했고,
캔·책상과 나머지 손/팔 collider의 rigid 접촉 및 기존 계수 보존, 복원 후 패드 유지, 240 physics tick의
유한 상태를 검사했다. 별도 0.1 kg 구의 단일 접촉 하중 시험에서 같은 pad material의 정적 압축은
0.04906 mm, rigid 지지면은 약 0.000005 mm였다. main 환경과 동일한 solver flag를 적용했을 때의 결과이며,
solver 기본값만 사용한 첫 진단에서는 예측 정적 변형과 맞지 않아 최종 진단 설정을 정정했다.
이 결과는 재질 적용·접촉법의 검증이고 실물 일치나 파지 성공 검증은 아니다.
상세 설정은 [contact.md](contact.md), 로컬 결과는 `local/reports/pad_materials/`에 둔다.

별도 팔 targets 재생에서 원래 rigid 재질로도 무효 자세가 재현됨을 확인했다. 해당 경로의
중복 Newton/PhysX mimic을 제거하는 것만으로는 해결되지 않았고, 같은 TGS solver와 반복수·scene flag를
적용한 뒤 625/625 sample을 완료했다. 관절 연결·기어비·drive gain은 유지한다. 이전 2000iter 정책은
strict 계약 검사와 새 접촉 물성의 재생은 통과했지만 111 control step에서 물체 오차로 종료되어
파지 성공을 주장하지 않는다. 새 물성으로 학습은 수행하지 않았다.

## DexYCB dataset attribution and raw-file tracking (2026-09-21)

두 사람 데모의 출처를 [DexYCB 공식 배포처](https://dex-ycb.github.io/)와 CVPR 2021 논문으로
명시했다. 원본 데이터의 CC BY-NC 4.0 표기와 프로젝트에서 수행한 리타게팅·좌표 보정·시간 조정을
[dataset.md](dataset.md)에서 구분한다. 확인되지 않은 demo2 전체 시퀀스 식별자는 추정하지 않는다.
원본 RGB·NPZ는 `data/demo*/raw/`에 로컬로 보존하고 Git 추적에서 제외한다. Git에는 배치 README와
출처/hash metadata, 실행에 필요한 소규모 파생 궤적을 둔다. 과거 `data/raw/`는 현재 tree에서 제거하며
이미 공유한 커밋 이력을 재작성하지 않는다. 이는 데이터 배포 범위의 변경이며 수치 입력 변경이 아니다.
