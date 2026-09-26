# Tasks

작업(task) 아래에 여러 데모를 등록합니다. 같은 번호라도 작업이 다르면 데이터와 정책을 별도로 선택합니다.

```text
config/tasks/
  can_pick/                  캔 집기 설정·데모 2
    task.json                작업 정의·실행 어댑터
    play.json                데모·정책 목록
    policy_demo*.json        학습·보상
    retargeting_demo*.json   리타게팅
    ik_demo*.json            팔 IK
  drilling/                  드릴 집기
    task.json
    play.json
    capture.json             RGB 데이터 제작 설정
src/dex_manipulation/tasks/   작업 등록·작업별 실행 구성
data/can_grasping/demo2/      can_pick 데모 2
data/drilling/demo1/          드릴 작업 입력
assets/tasks/drilling/        드릴 mesh·물리 자산
local/results/policy/<task>/  새 학습 결과
local/results/play/<task>/    재생 결과
```

FK·IK·리타게팅·PPO·로봇 모델은 공통 모듈을 사용합니다.
작업별 환경·보상·종료 조건은 해당 작업의 어댑터에서 연결합니다.

```bash
./run.sh tasks
./run.sh tasks drilling
./run.sh floating policy 2 --task can_pick
./run.sh train floating 2 --task can_pick -i 1000
```

`--task` 생략 시 `can_pick`입니다. 이어 학습은 checkpoint의 task를 따릅니다.
이전 checkpoint에서 task가 없으면 `can_pick`으로 해석하고 저장 파일은 수정하지 않습니다.

## Drilling 입력

첫 동작은 접근 → 파지 → 들어 올리기 → 유지입니다. 촬영 중 드릴은 하나의 rigid object로 취급하며 비트 회전·트리거 작동을 사용하지 않습니다.
나사 접근·정렬·트리거 누르기는 이후 확장할 별도 단계입니다. 드릴 집기 데이터에는 나사·지그·트리거 정보가 필요하지 않습니다.

RGB 데이터 제작은 `capture.json`을 사용합니다. 영상·mesh를 준비하는 경로와 변환 방법은 [RGB 데이터 제작](dataset_capture.md)을 참조합니다.
`task.json.inputs`에는 로봇 재생·학습에 사용할 드릴 자산, 데모, 장면, 리타게팅·IK·정책 설정을 정의합니다.
데모 제작과 로봇 실행 등록은 별개이며, 로봇 실행에는 해당 작업의 환경·보상 어댑터와 `play.json` 등록이 필요합니다.
입력이나 실행 어댑터가 없는 작업은 시뮬레이터를 시작하기 전에 중단합니다.

## 작업 추가

1. `config/tasks/<name>/task.json`과 `play.json`을 만듭니다. 이름은 영문 소문자·숫자·밑줄입니다.
2. `data/<name>/demoN/`, `assets/tasks/<name>/`에 입력과 형상을 둡니다.
3. 정책 설정에 `task_id`, `demo_id`를 넣고 `play.json`에 설정·입력 경로를 연결합니다.
4. `src/dex_manipulation/tasks/<name>.py`에 필요한 실행 구성을 작성하고 `task.json.entrypoints`에 등록합니다.

| Entry point | 호출 |
|---|---|
| `playback_plan` | `(root, args, catalog)` → 공통 재생 실행기가 소비할 계획 |
| `training_plan` | `(root, args)` → 공통 학습 실행기가 소비할 계획 |
| `policy` | `(args, root, on_ready=None, is_running=None)` → Isaac 시작 후 실행 |
| `reward` | `(env, metadata, training_reference)` → 해당 작업의 보상 연결 |

함수는 `dex_manipulation.tasks.<name>:function` 형식으로 지정합니다.
`can_pick.py`의 계획 반환 형식을 재사용하되 캔의 원통 충돌·다섯 손가락 접촉 보상을 다른 물체에 그대로 적용하지 않습니다.
작업 목록은 폴더를 자동 탐색하므로 목록을 관리하는 Python 분기를 추가할 필요가 없습니다.
