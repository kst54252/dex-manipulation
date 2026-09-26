# 재생

물체 pose 입력 없이 저장된 팔·손 궤적을 실행하려면 [고정 궤적 실행](execution.md)을 사용합니다.

```bash
./run.sh
```

메뉴에서 환경 → 리타게팅/정책 → 데모를 선택합니다. 파일 경로는 저장소 루트 기준입니다.

| 환경 | 리타게팅 | 정책 |
|---|---|---|
| 플로팅 Revo2 | `./run.sh floating retarget 2` | `./run.sh floating policy 2` |
| RB3 + Revo2 | `./run.sh arm retarget 2` | `./run.sh arm policy 2` |

캔 집기는 번호 `2`를 사용합니다. `retarget`는 손목·관절 기준 궤적을, `policy`는 학습 정책의 residual을 적용해 재생합니다.
`--task can_pick`으로 작업을 지정하며 생략해도 같은 작업을 사용합니다. 작업 목록과 추가 방법은 [task 구성](tasks.md)에 있습니다.
팔 리타게팅은 입력·배치에 맞는 IK 궤적을 준비하고, 팔 정책은 온라인 IK를 사용합니다.

## 옵션

| 옵션 | 설명 |
|---|---|
| `--repeat N` | 반복 횟수. 기본 `0`은 무한 반복 |
| `--speed N` | 학습/reference 기준 배속. 정책 1, 플로팅 리타게팅 2, 팔 리타게팅 1이 기본 |
| `--headless` | 창 없이 물리 재생 |
| `--dry-run` | 파일 생성 없이 실행 계획 출력 |
| `--list` | 등록 데모·정책 목록 |
| `--table-safety checkpoint` | 기본 상판 보호 대신 checkpoint의 설정 사용 |
| `--contact-materials checkpoint` | 기본 고무 패드 프로필 대신 학습 당시 물성 사용 |
| `--config PATH` | 정책 설정 직접 지정 |
| `--arm-config PATH` | 팔 배치·IK 설정 직접 지정 |

위치·관절은 선형 보간, 회전은 SLERP를 사용합니다. 배속을 바꿔도 물리 120Hz·제어 30Hz는 유지합니다.
창 닫기 또는 Ctrl+C로 종료합니다.

## 정책 선택

```bash
./run.sh floating policy 2
./run.sh arm policy local/results/policy/my_run/policy.pt
```

번호 `2`는 `local/results/policy/`의 데모2 최신 학습 완료 정책을 자동 선택합니다.
전체 학습 로그와 checkpoint의 iteration·설정이 일치하는 실행 중 저장 시각이 가장 최근인 것을 사용합니다.
학습 중·중단·손상된 실행은 제외합니다. 팔 학습 정책은 `arm` 환경에서만 선택합니다.
선택한 경로는 터미널과 `launch.json`에 표시됩니다.

직접 지정한 checkpoint나 등록된 고정 정책 이름은 자동 교체하지 않습니다.
checkpoint 옆의 `config.resolved.json`과 해당 모델·reference 파일을 함께 사용합니다.
별도 경로에 저장한 정책은 경로로 지정하거나 `local/results/policy/` 아래에 실행 폴더를 보관합니다.
`config/tasks/can_pick/play.json`에서 자동 선택 데모·고정 정책·기본 배속을 설정합니다.

## 데모2 랜덤 캔 배치

```bash
./run.sh arm policy 2 --random-can
./run.sh arm policy 2 --random-can --placement-seed 42 --repeat 3
```

매 회차 현재 정책의 궤적·재생 속도에 맞는 5cm IK 격자점을 섞어 선택합니다. 바로 직전 위치는 반복하지 않습니다.
손·캔의 XY를 함께 이동하고 높이·방향을 유지합니다. 팔 초기 관절값과 정책 좌표계도 함께 초기화합니다.

후보는 리타게팅 궤적의 IK·충돌·특이점·관절 한계 조건으로 선택합니다.
정책 residual은 실행 중 온라인 IK로 처리합니다. IK 실패 시 회차를 종료하고 다음 위치에서 시작합니다.
`--placement-seed`로 순서를 재현하며, 옵션이 없으면 고정 배치입니다.
지도 경로는 `config/tasks/can_pick/play.json`에서 관리합니다. 정책별 궤적 해시와 배속으로 `demos.2.random_can_policy_regions`를 선택하며, 기존 데모는 `random_can_regions`를 사용합니다.
전체 손·캔 궤적과 시간, 입력·USD·설정이 지도와 일치해야 실행할 수 있습니다.

## 출력

`local/results/play/<task>/`에 실행별 폴더를 만듭니다.
`launch.json`·`run_metadata.json`은 설정, `episodes.jsonl`은 회차 지표,
`first_episode.npz`는 첫 회차 상태·목표를 저장합니다.
랜덤 배치의 첫 회차 좌표변환은 `first_episode_placement.json`에 저장합니다.

## Tactile 기록

```bash
./run.sh arm policy 2 --record-tactile --repeat 1
./run.sh floating policy 2 --record-tactile --repeat 1
```

회차별 정상력·접선력·방향각은 실행 폴더의 `tactile/`에 저장합니다.
[센서 단위·시뮬레이션 근사·실물 기록](tactile.md)

실물: `./run.sh execute probe --hardware-config local/hardware.json --record-tactile`로 읽기 연결을 확인한 뒤,
`./run.sh execute hardware --hardware-config local/hardware.json --send --record-tactile`로 한 번 실행·측정합니다.
연결·현장 보정·초기 자세 준비는 [실물 절차](hardware_measurement.md)를 따릅니다.

관절 추종 제한 적용 전 정책은 `./run.sh arm policy 2-before-finger-tracking` 또는 `./run.sh floating policy 2-before-finger-tracking`으로 선택합니다. 기존 저장 궤적은 변경하지 않으며 새 녹화는 별도 경로를 사용합니다.

접촉 제약을 포함한 새 궤적 생성: [접촉점 리타게팅](retargeting.md).
