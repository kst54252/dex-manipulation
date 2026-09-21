# 간단한 시뮬레이터 재생

학습용 실행기는 `./train.sh`입니다. `./run.sh train`도 같습니다. [학습 명령](train.md).

저장소에서 `./run.sh`만 실행하면 **환경 → 동작 → 데모/정책** 선택 메뉴가 나옵니다.
현재 머신의 Isaac Python을 자동 선택하고 필요한 Python 경로를 설정합니다.
다른 디렉터리에서도 `/home/wanjunkim/ARSL/dex-manipulation/run.sh`로 실행할 수 있습니다.

```bash
./run.sh
```

## 바로 실행

| 환경·동작 | 1번 데모/정책 | 2번 데모/정책 |
|---|---|---|
| 플로팅 · 리타게팅만 | `./run.sh floating retarget 1` | `./run.sh floating retarget 2` |
| 로봇팔 · 리타게팅만 | `./run.sh arm retarget 1` | `./run.sh arm retarget 2` |
| 플로팅 · 정책 적용 | `./run.sh floating policy 1` | `./run.sh floating policy 2` |
| 로봇팔 · 정책 적용 | `./run.sh arm policy 1` | `./run.sh arm policy 2` |

기본은 **무한 반복**(`--repeat 0`)입니다. 플로팅·팔, 리타게팅·정책 모두 적용됩니다.
창 닫기 또는 Ctrl+C로 종료합니다. `--repeat 1`을 붙이면 한 번, `--repeat 3`은 세 번 재생합니다.
`--headless`에서도 기본은 무한 반복이며 Ctrl+C로 종료합니다.

1번은 가져온 40자세 동작(`data/demo1/`), 2번은 기존 27자세 동작(`data/demo2/`)입니다.
이전 번호를 서로 바꿨으며, 실행 이름 `original`은 `1`, `current`는 `2`의 호환 별칭입니다.
저장된 checkpoint의 옛 데이터 경로는 로더가 같은 동작의 새 위치로 연결합니다.

`retarget`는 학습 정책 없이 기존 물리 제어기로 전체 기하 궤적을 재생합니다. 중력·접촉이 적용되고
캔은 자유 강체입니다. 로봇팔의 IK 결과가 없거나 입력·모델·배치·제약 설정이 바뀌면 별도 경로에서
IK를 준비합니다. 실패/불연속 IK는 재생하지 않습니다. 리타게팅 입력이나 기존 검증 궤적을 덮어쓰지 않습니다.

`policy`는 선택한 checkpoint와 함께 저장된 `config.resolved.json`을 사용합니다. 학습 당시 reference와
저장된 물성을 복원하고 strict 관측, checkpoint 제어기, 한 환경에서 추론만 수행합니다.
정책이 종료 조건에 도달하면 첫 프레임부터 다음 회차를 시작합니다. 학습을 다시 시작하지 않습니다.

정책 재생에는 **손–상판 보호가 기본 적용**됩니다. 모든 손 collision mesh를 감싸는 링크별 상자로
상판 여유를 확인하고 위험한 손목 Z 목표만 올립니다. 관측한 링크 속도로 20ms 선행 여유를 두며,
명령의 최소 여유는 4mm입니다. 플로팅과 팔에 동일한 검사를 적용하고 팔 IK의 기존 제약을 유지합니다.
가중치 자체는 바뀌지 않으며, 목표가 달라지는 만큼 기존 정책의 물체 추종 성능도 달라질 수 있습니다.
이 보호는 동역학상 모든 충돌을 보장해서 막는 장치나 새 정책 학습의 대체물이 아닙니다.

기존 checkpoint 당시 동작을 비교하려면 다음과 같이 보호 설정까지 checkpoint에서 읽습니다.
새 상판 보상으로 학습한 checkpoint는 이 옵션에서도 학습 당시 보호가 유지됩니다.

```bash
./run.sh floating policy 2 --table-safety checkpoint
./run.sh arm policy 2 --table-safety checkpoint
```

실제로 적용한 설정은 `run_metadata.json`의 `execution.table_safety`, raw/applied 목표와
최소 여유·상향 보정량은 `first_episode.npz`에 저장됩니다. `retarget` 재생에는 이 정책 보호 옵션을 적용하지 않습니다.

**데모1 수평 교정:** `retarget 1`은 현재 `data/demo1/stable/reference.npz`를 사용합니다. 첫 캔 밑면이
수평이며 바닥 Z=0에 놓입니다. 반면 `policy 1`은 **수평 교정 전 2000iter 정책**을 재현하며,
첫 자세가 11.27° 기울어진 과거 입력을 사용한다는 안내가 나옵니다. 기존 정책의 목표·관측 계약을
새 데이터로 바꾸지 않습니다. 수평 교정된 입력은 [새 학습](train.md#데모1-수평-교정-후-학습)이 필요합니다.
학습 후 `./run.sh floating policy local/results/policy/demo1_flat_2000/policy.pt`로 재생합니다.
`arm`도 같은 checkpoint 경로를 받으며 현재 입력과 일치하는 `config/ik_demo1.json`을 자동 선택합니다.
팔 실행은 floating 정책을 온라인 IK로 연결하는 재생이며, 팔 환경에서 2000iter 학습한 정책을 뜻하지 않습니다.

1번의 실제 팔 학습 **20iter 시험 정책**은 `./run.sh arm policy 1-arm-trial`로 별도 선택합니다.
별도 데모가 아니라 1번 데모의 정책 변형이며, 현재 시험 정책은 파지에 실패했습니다.
이 시험 정책도 수평 교정 전 입력과 과거 `config/ik_original.json`을 사용합니다.

## 다른 정책 선택

```bash
./run.sh --list
./run.sh floating policy /경로/학습결과/policy.pt
./run.sh arm policy /경로/학습결과/policy.pt

# 같은 뜻의 명시적 옵션
./run.sh floating policy --policy 1

# checkpoint 옆에 설정이 없거나 새로운 데모의 팔 배치가 필요한 경우
./run.sh arm policy /경로/policy.pt \
  --config /경로/config.resolved.json --arm-config /경로/arm.json
```

직접 지정한 정책은 옆의 `config.resolved.json`을 읽습니다. 팔 설정은 등록된 정책과 정확히 같은 입력 파일을
사용하는 경우에만 자동 선택합니다. 새로운 입력은 `--arm-config`로 좌표 배치를 명시해야 합니다.
자주 쓰는 이름을 추가하려면 `config/play.json`의 `policies`에 `label`, `checkpoint`, `arm_config`를 등록합니다.

`arm_training.enabled`로 학습된 정책은 설정에 저장된 팔 배치를 자동 선택하고, 학습과 동일한
온라인 IK 및 실제 팔 제어 경로로 실행합니다. 상판 보호도 checkpoint 설정을 기본 사용합니다.
이 정책은 팔 상태 관측이 필요해 floating 환경에서는 실행할 수 없습니다. [팔 환경 학습](arm_policy.md).

## 확인용 옵션

```bash
./run.sh arm policy 1 --dry-run       # 실행·파일 생성 없이 연결할 입력/명령만 확인
./run.sh floating retarget 1 --headless --repeat 1
./run.sh --help
```

결과는 실행마다 `local/results/play/`의 별도 폴더에 저장합니다. `launch.json`에 실제 선택과 명령이 기록되며
학습 가중치는 덮어쓰지 않습니다. 물리 재생 완료와 물체 파지 성공은 별개입니다.

다른 머신의 Isaac Python은 `DEX_PYTHON=/절대경로/python ./run.sh`로 지정합니다. 기본 탐색은
`$HOME/IsaacLab/.venv/bin/python`, 프로젝트 `.venv/bin/python` 순서입니다. 이 실행기는 Isaac 환경을 설치하거나
현재 Python 환경을 변경하지 않습니다.

손끝의 다섯 `touch_link` 패드에만 rubber-like 접촉을 적용합니다. 캔·책상·나머지 손은 강체로
유지됩니다. 기존 정책도 기본 재생에서는 새 패드 물성을 사용하므로 학습 당시 접촉과 다릅니다.
`--contact-materials checkpoint`를 붙이면 해당 정책의 학습 물성을 재현합니다.
[계수·적용 부위·변경 전 설정](contact.md).
