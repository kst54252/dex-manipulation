# 학습

```bash
./run.sh train
./run.sh train floating 2 -i 1000 -n 4096
./run.sh train arm 2 -i 2000 -n 4096
./run.sh train floating 2 --task can_pick -i 1000
```

한 실행에서 한 데모를 학습합니다. 기본 4096환경, RSI·증강 사용, 창 없는 실행입니다.
`--task`는 작업을 선택합니다. 기본은 `can_pick`이며 새 결과는 `local/results/policy/<task>/`에 저장합니다.
학습 종료 후 재생은 [재생 명령](run.md)으로 직접 실행합니다. `train.sh`는 같은 실행기의 별칭입니다.

| 설정 | 역할 |
|---|---|
| `config/tasks/can_pick/policy_demo2_finger_tracking.json` | 기본 플로팅 학습 · 관절 추종 제한·패드 접근/접촉 보상 |
| `config/tasks/can_pick/policy_demo2_contact.json` | 기존 플로팅·패드 접근/접촉 설정 |
| `config/tasks/can_pick/policy_demo2_contact_only.json` | 데모2 기하 궤적·접촉력 보상만 사용 |
| `config/tasks/can_pick/policy_arm.json` | 실제 팔 상태와 온라인 IK를 포함한 학습 |
| `config/tasks/can_pick/play.json` | 데모별 입력·팔 설정 연결 |

다른 속도의 학습 입력은 [접촉 궤적 생성](contact_training.md)으로 준비합니다.
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
