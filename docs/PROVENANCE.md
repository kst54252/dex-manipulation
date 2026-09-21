# 방법론 출처

FK·기하 리타게팅은 논문과 USD·키포인트 정의로 독립 구현했습니다.
Residual RL은 REGRIND와 로컬 `regrind-revo2`의 방법·설정을 참고했습니다.
외부 task·환경 소스를 복사하거나 런타임에 import하지 않으며, PPO는 일반 RSL-RL 라이브러리를 사용합니다.

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
| `mdp/commands.py`, `mdp/rb3_revo2_commands.py` | reference-frame RSI, 자세·속도 reset, phase·속도 차분 | `policy/curriculum.py`, `reference.py`, `env.py` |
| `envs/events.py`, `dexterous_env_cfg.py` | 중력 단계, 물성·질량·관성·gain·COM randomization, 외란 | `policy/randomization.py`, `curriculum.py` |
| Revo2 floating config, `free_revo2_right_hand.py`, `tuna_can.py` | 120/30Hz, 손 중력 OFF, 손가락 drive, solver·접촉 설정 | `config/policy*.json`, `policy/env.py` |
| RSL-RL PPO config와 train/play 구성 | PPO·정규화·actor 초기화·checkpoint 흐름 | `policy/ppo.py`, `runner.py` |
| `regrind-revo2`의 접촉 reference 구성 | 물리 rollout의 손–물체 관계를 기준 궤적에 반영 | `policy/prepare.py` |

## 자체 설계

| 기능 | 구현 선택 |
|---|---|
| FK | 양쪽 joint frame과 Newton affine coupling 추출, NumPy FK, semantic 이름 대응 |
| 물체 점 | 삼각형 면적 비례 후보와 farthest-point sampling, 고정 50점 |
| 최적화 | SLSQP, translation+rotation-vector 손목, 이전 성공 해 warm start |
| 추가 추종 | 손 21점 위치와 손가락 segment 방향 항 |
| 충돌 | authored convex hull·원통의 FCL 거리와 보간 표본 제약 |
| 캔 | 사용자 지정 Ø73×32mm, 하단 Ø76×3mm 복합 원통 |
| 좌표 | 캔 밑면 Z=0, 데모1 수평화, 손·물체 공통 변환 |
| RB3 IK | USD 모델, adaptive DLS, branch 연속성, FK 오차·속도·특이점 조건 |
| 상판 보호 | 링크별 collision 경계 상자, 실제/목표 여유 보상, 손목 Z 보정 |
| 접촉 학습 | FK/FCL로 옆면 파지 입력 생성, 패드 접근·대향·동시 접촉 보상, preload |
| 팔 정책 | source-frame 관측, 가상 PD→IK 연결, batched IK를 포함한 팔 학습 |
| 재생 | 시간 재조정, 저장된 12-DoF 궤적 추종, IK 격자 내 회차별 랜덤 배치 |

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
