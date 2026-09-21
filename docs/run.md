# 재생

```bash
./run.sh
```

메뉴에서 환경 → 리타게팅/정책 → 데모를 선택합니다. 파일 경로는 저장소 루트 기준입니다.

| 환경 | 리타게팅 | 정책 |
|---|---|---|
| 플로팅 Revo2 | `./run.sh floating retarget 1` | `./run.sh floating policy 1` |
| RB3 + Revo2 | `./run.sh arm retarget 1` | `./run.sh arm policy 1` |

번호 `1`/`2`로 데모를 선택합니다. `retarget`는 손목·관절 기준 궤적을, `policy`는 학습 정책의 residual을 적용해 재생합니다.
팔 리타게팅은 입력·배치에 맞는 IK 궤적을 준비하고, 팔 정책은 온라인 IK를 사용합니다.

## 옵션

| 옵션 | 설명 |
|---|---|
| `--repeat N` | 반복 횟수. 기본 `0`은 무한 반복 |
| `--speed N` | 재생 배속. 플로팅·팔 정책 기본 2, 팔 리타게팅은 1 |
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

`config/play.json`에 정책 이름과 checkpoint를 등록합니다. 직접 지정한 checkpoint는 옆의
`config.resolved.json`을 읽습니다. 해당 설정의 모델·reference 파일도 함께 준비합니다.
팔에서 학습한 정책은 `arm` 환경을 사용합니다.

## 데모2 랜덤 캔 배치

```bash
./run.sh arm policy 2 --random-can
./run.sh arm policy 2 --random-can --placement-seed 42 --repeat 3
```

매 회차 5cm IK 격자점을 섞어 선택합니다. 2배속은 129개, 1배속은 133개이며 바로 직전 위치는 반복하지 않습니다.
손·캔의 XY를 함께 이동하고 높이·방향을 유지합니다. 팔 초기 관절값과 정책 좌표계도 함께 초기화합니다.

후보는 리타게팅 궤적의 IK·충돌·특이점·관절 한계 조건으로 선택합니다.
정책 residual은 실행 중 온라인 IK로 처리합니다. IK 실패 시 회차를 종료하고 다음 위치에서 시작합니다.
`--placement-seed`로 순서를 재현하며, 옵션이 없으면 고정 배치입니다.
지도 경로는 `config/play.json`의 `demos.2.random_can_regions`에 지정합니다.
지도·입력·USD·설정이 일치해야 실행할 수 있습니다.

## 출력

`local/results/play/`에 실행별 폴더를 만듭니다.
`launch.json`·`run_metadata.json`은 설정, `episodes.jsonl`은 회차 지표,
`first_episode.npz`는 첫 회차 상태·목표를 저장합니다.
랜덤 배치의 첫 회차 좌표변환은 `first_episode_placement.json`에 저장합니다.
