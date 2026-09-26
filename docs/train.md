# 학습

```bash
./run.sh train
./run.sh train floating 2 -i 1000 -n 4096
./run.sh train arm 2 -i 2000 -n 4096
./run.sh train floating 2 --task can_pick -i 1000
```

한 실행에서 한 데모를 학습합니다. 기본 4096환경, RSI·증강 사용, 창 없는 실행입니다.
`--task`는 작업을 선택합니다. 기본은 `can_pick`이며 새 결과는 `local/results/policy/<task>/`에 저장합니다.
학습 종료 후 재생은 [재생 명령](run.md)으로 직접 실행합니다.

| 설정 | 역할 |
|---|---|
| `config/tasks/can_pick/policy_demo2_finger_tracking.json` | 기본 플로팅 학습 · 관절 추종 제한·패드 접근/접촉 보상 |
| `config/tasks/can_pick/policy_demo2_contact.json` | 기존 플로팅·패드 접근/접촉 설정 |
| `config/tasks/can_pick/policy_demo2_contact_only.json` | 데모2 기하 궤적·접촉력 보상만 사용 |
| `config/tasks/can_pick/policy_arm.json` | 실제 팔 상태와 온라인 IK를 포함한 학습 |
| `config/tasks/can_pick/play.json` | 데모별 입력·팔 설정 연결 |

접촉 입력은 아래의 접촉 참조 생성 명령으로 준비합니다.
새 학습의 중력 일정은 선택한 설정과 iteration 수에 맞춰 조정됩니다.

## 옵션

| 옵션 | 역할 |
|---|---|
| `-i`, `--iterations` | 학습 횟수. 기본 floating 1000, arm 2000 |
| `-n`, `--num-envs` | 병렬 환경 수 |
| `--output PATH` | 새 `local/` 하위 결과 폴더 |
| `--save-every 100` | checkpoint 저장 주기 |
| `--config PATH` | 새 학습에 사용할 같은 데모·환경의 설정 |
| `--logger none` | TensorBoard 비활성화 |
| `--dry-run` | 학습 없이 설정·명령 확인 |

## 이어 학습

```bash
./run.sh train --resume local/results/policy/my_run/policy.pt -i 1000
./run.sh train arm 2 -i 2000 --initialize-actor local/results/policy/floating_run/policy.pt
```

`--resume`는 설정·optimizer·물성·커리큘럼을 복원하며 `-i`만큼 추가 학습합니다.
`--initialize-actor`는 호환되는 플로팅 actor와 관측 정규화를 새 플로팅/팔 학습의 초기값으로 사용합니다.
`--config`는 새 학습에만 적용하며 `--resume`와 함께 사용할 수 없습니다.

결과 폴더에는 `policy.pt`, `config.resolved.json`, `run_metadata.json`,
`training.jsonl`, `tensorboard/`가 저장됩니다. `policy.pt`와 사용한 reference를 함께 보관합니다.

## 손가락 과도한 오므림 제한

```bash
./run.sh train floating 2 -i 2000 -n 4096 \
  --config config/tasks/can_pick/policy_demo2_finger_tracking.json \
  --initialize-actor local/results/policy/demo2_half_capture_2000_20260922_120939/policy.pt
```

관절별 실측각 기준 ±3° 목표 제한과 USD 목표 속도 제한을 적용합니다. 이 설정은 강성 9 N·m/rad, 감쇠 0.173 N·m·s/rad, 기존 토크 상한 0.5 N·m를 사용하며 중력을 전체 학습의 25% 지점까지 올립니다. 제한 전 과도한 명령과 실제 추종 오차를 벌점으로 주며, 기존 물체 추종·접촉 보상은 유지합니다. 두 목표 생성기가 겹치지 않도록 `motion_control`은 끕니다.
변경된 보상으로 학습할 때는 `--initialize-actor`를 사용합니다. `--resume`는 기존 설정·optimizer까지 그대로 이어가는 옵션입니다.
기존 정책·명령 파일은 수정하지 않습니다. 새 정책의 실물용 궤적은 별도 `execute record` 후 `execute replay`로 파지와 관절 오차를 확인합니다.

## 정책 구조

정책은 기준 궤적에 손목 위치 3·회전 3·독립 손가락 관절 6개의 residual을 더합니다.
플로팅 손목은 자세 PD로, 손가락은 관절 drive로 제어합니다. PPO는 RSL-RL을 사용합니다.

| 구성 | 동작 |
|---|---|
| 관측 | 손·물체 상태, 기준 궤적, phase, 이전 action; actor 67 / critic 94차원 |
| 보상 | 물체 50점 추종, 선·각속도, 손목 자세, action 변화·크기 |
| RSI | 마지막 프레임을 제외한 기준 제어 프레임을 뽑아 자세·속도 초기화 |
| 증강 | XY·yaw 변환, 관측 noise·delay, 물성·질량·gain·관절 기본값 randomization |
| 중력 | 단계별 커리큘럼; 손 링크 중력 OFF, 캔 중력 적용 |
| 제어 주기 | 물리 120Hz, 정책 30Hz |
| 상판 보호 | 링크별 collision 경계 상자의 여유 계산과 손목 Z 보정 |
| 접촉 학습 | 데모2: 접근·대향·동시 다섯 접촉 보상 |

상판 보상은 실제 손과 보호 전 목표의 여유를 각각 계산합니다.
기본 명령 여유는 4mm, 속도 선행 시간은 20ms입니다.
패드 접촉 설정은 [접촉 물성](environment.md), 접촉 학습은 아래 접촉 보상 절에 있습니다.


## 팔 환경

손목·손가락 residual을 RB3 + Revo2 고정 베이스 articulation에서 실행합니다.
손목 목표는 온라인 IK로 팔 관절 명령으로 변환합니다.

| 항목 | 플로팅 정책을 팔에 연결 | 팔 환경에서 학습 |
|---|---|---|
| 출력 | 손목 6 + 손가락 6 residual | 동일 |
| Actor / critic | 67 / 94차원 | 87 / 114차원 |
| 손목 제어 | 가상 PD 응답 → CPU IK | 상판 보호 → batched GPU IK |
| 팔 상태 관측 | 손목·손의 실제 상태 | arm q·qdot, IK 잔차·성공·실패 횟수 추가 |
| 팔 관련 보상 | 기본 손·물체 보상 | IK·추종·특이점·관절 한계 비용 추가 |

IK는 이전 해를 시작점으로 삼고 실제 dt의 관절 속도·step 제한을 적용합니다.
위치·회전을 함께 맞춘 뒤 FK 오차와 Jacobian을 검사합니다.
실패 시 직전 관절 목표를 유지하며, 팔 학습에서는 연속 실패 횟수로 종료합니다.


## 패드 접근·접촉 학습

데모2는 다섯 패드가 캔 옆면에 접근하는 기준 궤적과 접촉 보상을 사용합니다.
물체 궤적·시간·사람 키포인트는 유지하며 손끝에는 고무 패드 물성을 적용합니다.

| 보상 | 역할 |
|---|---|
| 패드 표면·가장 먼 패드 거리 | 접촉 전 손가락 전체의 접근 유도 |
| 엄지 대향·다섯 패드 접촉 | 같은 환경의 PhysX 패드–캔 힘으로 접촉 유지 |
| 접촉력 | 작은 압착 유지, 과도한 힘 억제 |
| 물체 추종 | 궤적 오차에 따라 접촉 보상 조절 |

원본 frame 26부터 마지막까지 각 120Hz 물리 substep의 접촉력을 측정합니다.
접촉 기준은 0.05N, 과도한 힘 기준은 패드당 5N입니다.
기본 설정은 파지 직전 손가락 목표에 작은 preload를 점진 적용합니다.

## 입력과 설정

프로젝트를 설치한 Python 환경에서 실행하며 출력은 새 파일을 지정합니다.

```bash
./run.sh data contact-reference --source data/can_grasping/demo2/grounded/reference.npz \
  --output local/references/contact/demo2.npz --start-frame 26 --blend-start .8 --side-contact
./run.sh train floating 2 -i 1000
```

FK/FCL·SLSQP로 관절·속도·캔·상판 조건을 계산하고 입력 hash와 생성 설정을 기록합니다.
`config/tasks/can_pick/policy_demo2_contact.json`은 접근·접촉 보상과 preload를 함께 사용합니다.
기하 궤적을 유지하고 접촉력 보상만 사용하려면 다음 설정을 선택합니다.

```bash
./run.sh train floating 2 --config config/tasks/can_pick/policy_demo2_contact_only.json -i 1000
./run.sh floating policy 2
```

기존 접촉력 전용 checkpoint도 같은 `run.sh` 학습·재생 경로를 사용합니다.
`grasp_reference.py`는 입력 생성, `policy/grasp_reward.py`는 접근 보상·preload,
`policy/contact_reward.py`는 접촉력 보상을 담당합니다.

## 공통 설정 상속

정책 설정은 `"$extends": "policy_demo2.json"`처럼 같은 폴더의 공통 설정을 상속합니다. 객체는 항목별로 병합하고 배열·숫자는 자식 값으로 교체합니다. Python에서는 `configuration.read_config()`로 읽습니다. 학습 결과의 `config.resolved.json`에는 상속을 해석한 전체 설정을 저장합니다.
