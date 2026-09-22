# 학습

```bash
./run.sh train
./run.sh train floating 1 -i 1000 -n 4096
./run.sh train floating 2 -i 1000 -n 4096
./run.sh train arm 2 -i 2000 -n 4096
```

한 실행에서 한 데모를 학습합니다. 기본 4096환경, RSI·증강 사용, 창 없는 실행입니다.
학습 종료 후 재생은 [재생 명령](run.md)으로 직접 실행합니다. `train.sh`는 같은 실행기의 별칭입니다.

| 설정 | 역할 |
|---|---|
| `config/policy_demo1.json` | 데모1 플로팅·손가락 접촉 보상 없음 |
| `config/policy_demo2_contact.json` | 데모2 플로팅·패드 접근/접촉 보상 |
| `config/policy_demo2_contact_only.json` | 데모2 기하 궤적·접촉력 보상만 사용 |
| `config/policy_arm.json` | 실제 팔 상태와 온라인 IK를 포함한 학습 |
| `config/play.json` | 데모별 입력·팔 설정 연결 |

데모1은 바닥에 정렬된 기존 리타게팅 입력을 사용합니다.
데모2의 기본 학습 입력은 [접촉 궤적 생성](contact_training.md)으로 준비합니다.
새 학습의 중력은 전체 iteration의 75% 지점에서 9.81 m/s²에 도달합니다.

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
./run.sh train arm 1 -i 2000 --initialize-actor local/results/policy/floating_run/policy.pt
```

`--resume`는 설정·optimizer·물성·커리큘럼을 복원하며 `-i`만큼 추가 학습합니다.
`--initialize-actor`는 호환되는 플로팅 actor와 관측 정규화를 새 팔 학습의 초기값으로 사용합니다.
`--config`는 새 학습에만 적용하며 `--resume`와 함께 사용할 수 없습니다.

결과 폴더에는 `policy.pt`, `config.resolved.json`, `run_metadata.json`,
`training.jsonl`, `tensorboard/`가 저장됩니다. `policy.pt`와 사용한 reference를 함께 보관합니다.
