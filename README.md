# dex-manipulation

사람 손·물체 데모를 Revo2 동작으로 리타게팅하고, RB3-730 팔 IK와 residual RL을 연결하는 Isaac Sim 프로젝트입니다.
작업별 설정·데모·정책은 task로 구분하며 기본 작업은 `can_pick`입니다.

```text
DexYCB / 직접 촬영한 RGB → 손 21점 + 물체 6D pose
  → 손목·손가락 리타게팅
  → 플로팅 Revo2 / RB3 + Revo2
  → 물리 재생·정책 학습
```

## 실행

Git clone에는 데모2 최신 정책과 실행 입력도 포함됩니다. `run.sh`가 `runtime/`의 검증된 묶음을
Git 제외 영역인 `local/`에 자동 복원하며 기존 파일은 덮어쓰지 않습니다.

```bash
git clone https://github.com/kst54252/dex-manipulation.git
cd dex-manipulation
./run.sh check                                       # 파일 복원·SHA256 검사
./run.sh setup sim /Isaac환경/bin/python              # 설치된 Isaac Python에 프로젝트 의존성 설치
# 실물 저장 궤적 실행·촉각 측정만 필요한 PC:
./run.sh setup hardware
```

시뮬레이션 실행 기준은 **Isaac Sim 6.0.1 / Python 3.12**입니다. Isaac Sim·GPU 드라이버는 별도로 설치합니다.
`run.sh`는 `~/IsaacLab/.venv` 또는 저장소의 `.venv`를 찾습니다.
다른 환경은 `DEX_PYTHON=/경로/python ./run.sh`로 지정합니다. [다른 PC 설치](docs/installation.md)

```bash
./run.sh                      # 환경·데모·정책 선택 메뉴
./run.sh floating retarget 2   # 플로팅 손 · 리타게팅
./run.sh arm retarget 2        # 로봇팔 연결 · 리타게팅
./run.sh floating policy 2     # 플로팅 손 · 학습 정책
./run.sh arm policy 2          # 로봇팔 연결 · 학습 정책
./run.sh tasks                 # 작업 목록
./run.sh tasks drilling        # 드릴 작업의 구성·필요 입력
./run.sh arm policy 2 --task can_pick
```

캔 집기는 **데모2**를 사용합니다. 정책은 `local/results/policy/`에서 데모2의 최신 학습 완료 checkpoint를 자동 선택합니다.
학습 중·중단된 실행은 제외하며 실행 시 선택한 경로를 표시합니다. 특정 정책은 checkpoint 경로로 지정합니다.
체크포인트·접촉 학습 입력·IK 지도는 로컬 `local/`에 보관합니다.

| 옵션 | 동작 |
|---|---|
| `--repeat 3` | 3회 재생. 기본은 무한 반복, Ctrl+C 또는 창 닫기로 종료 |
| `--speed 1` | 학습/reference 시간 그대로. 정책 기본 1, 플로팅 리타게팅 2, 팔 리타게팅 1 |
| `--headless` | 창 없이 실행 |
| `--dry-run` | 실행할 입력과 명령 확인 |

```bash
./run.sh arm policy 2 --random-can             # 매 반복 IK 영역에서 캔 위치 변경
./run.sh floating policy local/results/policy/my_run/policy.pt
```

## 학습

```bash
./run.sh train                                # 학습 메뉴
./run.sh train floating 2 -i 1000 -n 4096
./run.sh train arm 2 -i 2000 -n 4096
```

학습은 창 없이 실행하며 결과를 `local/results/policy/`에 저장합니다.
`train.sh`도 같은 학습 실행기입니다. [재생 옵션](docs/run.md) · [이어 학습·설정](docs/train.md)

## 프로젝트 구성

| 경로 | 내용 |
|---|---|
| `assets/` | RB3·Revo2 USD, 캔 형상, 키포인트, 추출 모델 |
| `data/can_grasping/demo2/` | 사람 손·물체 데모와 리타게팅 궤적 |
| `data/<task>/demoN/` | 추가 작업의 촬영 입력·변환 데이터 |
| `config/tasks/<task>/` | 작업별 데모·환경·IK·정책 설정 |
| `config/` | 공통 로봇 실행·통신·작업대 설정 |
| `src/dex_manipulation/` | FK, IK, 리타게팅, 시뮬레이션, 정책 패키지 |
| `scripts/` | `run.sh`에서 사용하는 실행 진입점 |
| `docs/` | 기능과 설정 설명 |
| `runtime/` | 최신 정책·필수 참조·실물 명령·IK 지도 배포 묶음과 SHA256 목록 |
| `local/` | Git 제외: 실행 결과·체크포인트·테스트·분석·공유 파일 |

코드는 다음 기능별로 나뉩니다.

```text
src/dex_manipulation/
├── fk.py · ik.py · retargeting.py   # 손·팔 운동학과 리타게팅
├── geometry.py · scene.py          # 형상·충돌·작업대
├── policy/                        # 물리 환경·보상·PPO 학습·재생
├── robot/                         # 저장 궤적·실물 장치·ROS·가상 제어·표시
├── sensors/                       # tactile 수집·그래프·비교
├── dataset/                       # RGB 영상의 손·물체 pose 복원
└── tasks/                         # can_pick·drilling 작업 등록과 어댑터
```

정책 설정은 `policy_demo2.json`을 상속하고 다른 값만 명시합니다.
학습 시 전체 설정을 `config.resolved.json`으로 저장하므로 기존 정책과 궤적의 계약을 유지합니다.

```bash
./run.sh data --help           # 추출·가져오기·바닥 정렬·캔/접촉 참조 생성
./run.sh retargeting --help    # USD 추출·리타게팅·비교 뷰어
./run.sh ik --help             # 팔 모델 추출·IK 궤적 생성
./run.sh tracking --help       # 저장 정책 궤적의 팔 추종
```

[데이터](docs/dataset.md) · [리타게팅](docs/retargeting.md) · [IK](docs/ik.md) ·
[정책·학습](docs/train.md) · [환경·물성](docs/environment.md) · [작업 추가](docs/tasks.md)

## RGB 데모 제작

`drilling`의 첫 동작은 드릴에 접근해 잡고 들어 유지하는 동작입니다. RGB 영상과 실제 크기를 아는 드릴 mesh를 사용하며 RGB-D는 필요하지 않습니다.
카메라·책상 좌표 보정, 손/물체 pose 복원, EgoPHI 접촉 추정, 검수와 리타게팅 입력 내보내기를 분리합니다.

```bash
./run.sh dataset init 1 --task drilling
./run.sh dataset status 1 --task drilling
./run.sh dataset --help
```

영상·mesh·카메라 보정값을 추가한 뒤 [RGB 데이터 제작](docs/dataset_capture.md)의 순서로 처리합니다.
HaMeR/EgoPHI는 별도 환경·체크포인트를 사용합니다. 단안 손 pose와 힘 출력은 측정 정답과 구분해 저장합니다.

팔·손 통합 조작과 실시간 표시에는 [ROS 2 연동](docs/robot.md)을 사용합니다.

```bash
./run.sh robot virtual    # 가상 관절 조작 패널 + RViz
./run.sh robot sim        # Isaac 물리 시뮬레이션 + RViz
./run.sh robot hardware   # 실물 관절 상태 수신 + RViz (읽기 전용)
```

실물 명령·보정 설정과 제조사 Virtual Control Box 연결은 [통합 제어 안내](docs/robot.md)를 참조합니다.

기존 캔 데모의 사람 손·물체 데이터는 **[DexYCB](https://dex-ycb.github.io/)**를 사용합니다.
원본 이미지·annotation은 로컬에 보관합니다. [데이터 출처](docs/dataset.md) · [방법론 출처](docs/PROVENANCE.md)
