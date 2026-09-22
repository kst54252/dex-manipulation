# Tasks

작업(task) 아래에 여러 데모를 등록합니다. 같은 번호라도 작업이 다르면 데이터와 정책을 별도로 선택합니다.

```text
config/tasks/
  can_pick/                  캔 집기 설정·데모 1, 2
    task.json                작업 정의·실행 어댑터
    play.json                데모·정책 목록
    policy_demo*.json        학습·보상
    retargeting_demo*.json   리타게팅
    ik_demo*.json            팔 IK
  drilling/                  드릴 집기·나사 정렬
    task.json
    play.json
src/dex_manipulation/tasks/   작업 등록·작업별 실행 구성
data/demo1/, data/demo2/      기존 can_pick 입력·checkpoint 경로 유지
data/drilling/demo1/          드릴 작업 입력
assets/tasks/drilling/        드릴·나사·지그 USD
local/results/policy/<task>/  새 학습 결과
local/results/play/<task>/    재생 결과
```

FK·IK·리타게팅·PPO·로봇 모델은 공통 모듈을 사용합니다.
작업별 환경·보상·종료 조건은 해당 작업의 어댑터에서 연결합니다.

```bash
./run.sh tasks
./run.sh tasks drilling
./run.sh floating policy 2 --task can_pick
./run.sh train floating 1 --task can_pick -i 1000
```

`--task` 생략 시 `can_pick`입니다. 이어 학습은 checkpoint의 task를 따릅니다.
이전 checkpoint에서 task가 없으면 `can_pick`으로 해석하고 저장 파일은 수정하지 않습니다.

## Drilling 입력

동작은 드릴 집기 → 나사 접근 → 비트 정렬 → 트리거 누르기 → 접촉 유지로 구분합니다.
정책은 손목·손가락을 제어하며, 비트 회전은 트리거를 누르는 물리적 상호작용으로 발생시킵니다.

`task.json`의 `inputs`에 드릴·지그 형상, 사람 데모, 비트/나사/트리거 좌표계와 물성 입력 경로를 정의합니다.
실제 회전 속도·토크·트리거 조건, 정렬·접촉 성공 조건을 설정한 뒤 실행 어댑터와 `play.json` 데모를 등록합니다.
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
