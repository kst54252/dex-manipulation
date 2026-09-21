# dex-manipulation

Revo2 손 FK → 사람 손/물체 리타게팅 → RB3 arm IK / residual RL 학습·평가 프로젝트입니다.

사람 손·물체 데모는 **[DexYCB](https://dex-ycb.github.io/) (Chao et al., CVPR 2021)**를 사용합니다.
원본 RGB·annotation은 로컬에 보관하고 Git에서 제외합니다.
[데이터 출처·라이선스·데모별 처리 내용](docs/dataset.md).

시뮬레이터 재생은 저장소에서 **`./run.sh`**를 실행하고 메뉴를 선택하면 됩니다. 환경 활성화나
`PYTHONPATH` 입력 없이 실행합니다. 바로 실행하려면 다음 네 명령을 사용하세요.

```bash
./run.sh floating retarget 1   # 플로팅 · 리타게팅만
./run.sh arm retarget 1        # 로봇팔 · 리타게팅만
./run.sh floating policy 1     # 플로팅 · 학습 정책
./run.sh arm policy 1          # 로봇팔 · 학습 정책
```

**1번 데모**는 가져온 40자세 동작, **2번 데모**는 기존 27자세 동작입니다. 사용자 요청으로 이전 번호를 서로 바꿨습니다.
마지막 `1`을 `2`로 바꾸면 2번 데모/정책을 선택합니다. 기본은 **무한 반복**이며
창 닫기 또는 Ctrl+C로 종료합니다. `--repeat 1`을 붙이면 한 번만 재생합니다. 다른 정책은 마지막 인자에 `.pt` 경로를 넣습니다.
[정책 선택·실행 옵션](docs/run.md). 데모1의 새 학습·리타게팅 입력은 **캔 밑면을 수평으로 만들어
바닥에 밀착시킨 5.2초 궤적**입니다. 등록된 `policy 1`은 이 수평 교정 전 2000iter 정책이며,
교정된 좌표계에서는 새 학습이 필요합니다. 실행 메뉴에도 이 차이를 표시합니다.

학습은 **`./train.sh`** 메뉴 또는 아래 명령을 사용합니다. 마지막 `1`을 `2`로 바꾸면
2번 데모입니다. 기본 4096환경에서 창 없이 학습만 수행하며 종료 후 평가·재생을 열지 않습니다.

```bash
./train.sh floating 1 --iterations 2000
./train.sh arm 1 --iterations 2000
```

수평 교정한 데모1의 플로팅 학습과 학습 후 재생:

```bash
./train.sh floating 1 -i 2000 --output local/results/policy/demo1_flat_2000
./run.sh floating policy local/results/policy/demo1_flat_2000/policy.pt
```

새 학습의 중력은 전체 횟수의 75% 지점에서 9.81 m/s²에 도달합니다.
`./run.sh train ...`도 같은 명령입니다. [이어 학습·저장 경로·실행 확인](docs/train.md).

```text
assets/                  로봇·캔 원본 USD, 키포인트 정의, 추출 모델
  can/                    현재 복합 원통·50점·치수, original/에 기존 캔 보관
  models/                Isaac 없이 FK에 사용하는 JSON/NPZ
config/
  retargeting.json        데이터 좌표계·물체 대응·시간 설정
  policy.json             RL 환경·보상·학습, 플로팅 30×30cm 지지면 설정
  ik.json                 RB3 모델 프레임·IK 제약 설정
  workcell.json           팔 환경의 책상·받침대 치수, 로봇 장착 위치 (책상 윗면 z=0)
data/                    두 데모의 입력 데이터
  demo1/                  1번 데모: 12~51번, 40자세, 5.2초
  demo2/                  2번 데모: 15~41번, 27자세, 5.2초
src/dex_manipulation/      재사용 Python 패키지
  fk.py                   손·팔 관절값 → 링크·키포인트
  ik.py                   손목 pose → RB3 6축 IK
  reference.py            순수 12-DoF joint target·보간·CSV
  sim.py                  Isaac articulation adapter
  scene.py                공통 책상·받침대 생성, world↔robot base 변환
  control.py              플로팅 손목 PD·링크별 토크 분배
  object.py               치수로 캔 메시·충돌 형상·키포인트 생성
  retargeting.py          사람 동작 → 로봇 손목·관절 궤적
  policy/                 residual RL 환경·PPO·학습·평가
  data.py                 입력 검증·semantic 대응·시간 처리
  usd.py                  USD에서 모델 추출
  geometry.py             물체 표면점·collision geometry
  mesh.py                 interaction graph·Laplacian
  metrics.py              위치·방향 오차 계산
  transforms.py           좌표·회전 변환
  coordinates.py          데이터셋 전체의 바닥 좌표계 변환
  viewer.py               리타게팅 비교 재생
  cli.py                  리타게팅 명령 처리
  static/                 배포용 뷰어 번들
scripts/                  object.py / data.py / ground.py / retargeting.py / ik.py / replay.py / physics.py / policy.py / tracking.py
run.sh                    재생 메뉴·짧은 실행 명령 (scripts/run.py)
train.sh                  데모 2/2 · 플로팅/팔 학습 메뉴 (run.sh train과 동일)
web/                      비교 뷰어 JavaScript 소스
docs/                    사용법·방법론 출처
local/                    Git 제외: 개발 도구·테스트·분석·출력·공유본
  tests/                  단위 테스트와 합성 입력
  tools/                  Isaac 검증·문제 분석·공유 패키징 도구
  reports/                분석 문서·검증 로그·구조 변경 기록
  results/                실행 결과·HTML·학습 로그·체크포인트
  exports/                전달용 USD·ZIP
```

`local/` 전체는 `.gitignore`로 일반적인 `git add`/커밋 대상에서 제외합니다. 핵심 연산은 `local/`의 테스트 코드에 의존하지 않습니다. 실행 결과는 `local/results/`에 저장하며 재생기는 지정한 결과 파일을 읽습니다. 각 데모의 `poses.npz`는 사람 손·물체의 카메라 좌표, `retargeted.npz`는 카메라 좌표의 로봇 궤적, `grounded/reference.npz`는 바닥 좌표의 현재 기하 궤적입니다. 기존 정책의 정확한 학습 입력은 각 `policy_reference.npz`로 구분합니다. [두 데모의 구성과 실행 설정](docs/data.md)을 참고하세요. RB3 IK는 `ik.py`에 있으며 simulator 코드와 분리되어 있습니다.

## 실행

현재 캔은 **몸통 Ø73 mm / 전체 높이 32 mm / 하단 3 mm만 Ø76 mm**입니다. 키포인트 50개와 리타게팅·바닥 데이터·팔 IK를 새 형상으로 재생성했습니다. 기존 캔은 `assets/can/original/`, 교체 전 결과는 `local/archive/before_can_resize/`에 있습니다. [캔 형상·좌표계·재생성 방법](docs/object.md)을 참고하세요.

저장소 루트에서 실행합니다. 일반 환경은 `pip install -e .`로 패키지를 설치합니다. RL은 Isaac Sim·PyTorch·RSL-RL이 설치된 Python을 사용합니다. 현재 머신:

```bash
export PYTHON=/home/wanjunkim/IsaacLab/.venv/bin/python
export PYTHONPATH="$PWD/src:$PWD/.deps:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

```bash
# 데이터 추출 (기존 원본 RGB/label은 수정하지 않음)
"$PYTHON" scripts/data.py

# 리타게팅
"$PYTHON" scripts/retargeting.py retarget \
  --model-dir assets/models --input data/demo2/poses.npz \
  --metadata config/retargeting.json --output local/results/retargeting/839512060362

# 비교 재생 HTML 생성
"$PYTHON" scripts/retargeting.py view \
  --model-dir assets/models --result local/results/retargeting/839512060362 \
  --reference-dir data/demo2/raw

# 물리 재생: 학습 없이, 파지 실패 후에도 전체 동작 확인
"$PYTHON" scripts/physics.py floating --loops 3
"$PYTHON" scripts/physics.py arm --loops 3

# 학습: 4096 환경 / 1000 iterations, RSI ON, 중력 curriculum·augmentation ON
# GUI 없이 학습하고 종료 후 평가·재생도 실행하지 않음
"$PYTHON" scripts/policy.py --mode train --headless --skip-evaluation \
  --output local/results/policy/table_safe_1000
```

플로팅 학습·평가·물리 재생은 **30×30cm 평면 중앙 XY=(0,0)**에서 캔이 시작합니다. `config/policy.json`의 `surface`에서 크기를 지정하며 윗면은 **z=0**, 충돌용 두께는 2cm입니다. 책상 다리·로봇 받침대·하단 바닥은 생성하지 않습니다. 학습의 기존 XY 위치 증강(±5cm)은 유지합니다.

현재 RL 설정은 v8의 직접 residual 제어·손가락 drive·접촉·RSI에 자체 상판 여유 보상과 명령 보호를 추가한 버전입니다.
학습 reference는 자체 물리 rollout으로 접촉 자세를 준비한 `data/demo2/policy_reference.npz`이며 원래 캔 목표·시각·사람 손 점은 유지합니다. 기본 설정의 `local/policy/reference.npz`는 이 파일의 상대 링크입니다. 기존 체크포인트의 설정 해시를 보존하기 위해 설정 문자열은 유지합니다. [준비 명령과 출처](docs/policy.md)를 확인하세요.
RSI는 terminal frame을 제외한 제어 프레임을 균등하게 선택하고 손·물체의 자세와 속도를 함께 초기화합니다.
관측은 actor 67 / critic 94차원이며, 달라진 phase·속도·물리 규약 때문에 기존 checkpoint를 새 설정으로 이어 학습하지 않습니다.
기존 정책은 해당 `config.resolved.json`과 `--motion-control checkpoint --table-safety checkpoint`로 재현합니다.
새 설정의 추가 명령 governor는 기본 OFF입니다. 변경 내용과 남긴 차이는 [정책 학습](docs/policy.md)에 있습니다.

팔 환경은 `config/workcell.json`의 기존 책상 80×160cm, 중심 XY=(0.65,0)m, 상판 z=0을 사용합니다. 받침대는 50×50×70cm, 장착면 z=-0.02m, 바닥 z=-0.72m입니다. 캔은 사용자 지정대로 베이스 중심에서 수평 55cm 앞, 초기 XY=(0.55,0)m에 놓으며 **시계방향 90°** 방향을 유지합니다. `regrind-revo2`에서는 이 환경 치수·배치만 참고했습니다. 데이터·손 관절·시연 시간은 유지하며, 캔은 재생 중 자유 강체입니다. 자세한 좌표와 출처는 [환경 설정](docs/scene.md)에 있습니다. 실측 카메라–로봇 보정은 아닙니다.

현재 구간은 **15~41번, 27자세**이며 마지막 9프레임(42~50번)을 제외했습니다. `config/retargeting.json`의 `frame_range`가 재추출·리타게팅·바닥 좌표 생성에도 적용됩니다. 제외한 원본은 Git 제외 영역 `local/archive/before_tail_trim/`에 보관합니다.

팔 IK는 27자세 사이에 26자세를 보간해 총 53개를 계산합니다. 기존 5Hz 간격을 유지하여 팔/플로팅 물리 재생은 5.2초입니다. RL도 입력의 5.2초를 유지하고 30Hz에서 보간하며 10초 timeout과 분리합니다. 20° 관절 이동 제한과 속도 제한은 유지합니다.

플로팅 손은 로컬 REGRIND처럼 **손의 모든 링크 중력 OFF, 캔 중력 9.81 m/s² ON**입니다. `config/policy.json`의 `hand_gravity: false`를 미리보기와 RL에서 함께 사용합니다. 손목 자세 PD의 힘은 `right_hand_base_link`에, 토크는 링크 질량 비율로 나누어 적용합니다. 별도의 중력 보상·목표 속도 feedforward는 제거했습니다. 물리 120Hz/목표 갱신 30Hz를 유지하고, 화면 갱신은 추가 물리 스텝 없이 실행합니다. 손의 질량·관성·접촉은 유지되므로 동작 중 추종 지연이나 접촉 떨림은 남을 수 있습니다. 팔 결합 모델의 중력과 USD drive는 기존 설정을 사용합니다.

선택 기능인 `motion_control`은 30Hz 목표를 120Hz의 연속 명령으로 바꿉니다. 현재 학습에서는 원본처럼 OFF입니다.
`policy.py --mode play`와 간단 실행기의 정책 모드는 학습 당시 제어에 상판 보호를 기본 적용합니다.
충돌 형상으로 손 전체 여유를 검사하며 손목 Z 목표만 필요할 때 올립니다. 새 보상 학습은 별도로 필요합니다.
기존 정책의 보호 전 동작 비교에는 `--table-safety checkpoint`를 사용합니다. [설정과 검증 범위](docs/policy.md).
학습 정책을 RB3+Revo2에서 실행하려면 `policy.py --mode play --robot arm`을 사용합니다.
실제 팔/손/캔 상태로 정책을 계산하고 손목 출력을 strict IK로 변환합니다. 최신 checkpoint를 지정하는 명령과
학습 환경과의 차이는 [팔 결합 정책 재생](docs/policy.md#팔에-연결한-학습-정책-재생)에 있습니다.
팔 모드는 매 명령에 RB3 IK·관절 제약 검사를 적용하며, 플로팅 안정화가 팔 도달 가능성을 보장하지 않습니다.

원본 reference를 변경했거나 새 clone에서 팔 결과를 생성할 때:

```bash
"$PYTHON" scripts/ground.py
"$PYTHON" scripts/ik.py solve
```

보정된 사람 손/물체 데이터는 `data/demo2/grounded/poses.npz`, 실제 변환과 원본 hash는 `data/demo2/grounded/frame.json`에 있습니다. 원본 RGB 비교 뷰어는 표시할 때만 역변환해 원본 카메라와 맞춥니다.

자세한 명령과 입력·출력 규약: [리타게팅](docs/retargeting.md), [RB3 IK·12-DoF reference](docs/ik.md), [정책 학습](docs/policy.md), [방법론 출처](docs/PROVENANCE.md).
개발 도구 사용법은 로컬의 `local/README.md`에 있습니다. `local/`은 새 clone에 포함되지 않습니다.
앞으로의 커밋은 Conventional Commits 형식을 사용합니다. 커밋 규칙과 로컬 파일 관리 지침은 [AGENTS.md](AGENTS.md)에 기록했습니다.

## 공유 파일

공유용 파일도 `local/exports/`에 모았습니다. 마지막으로 만든 단일 파일은
`local/exports/revo2_standalone_2026-09-19/revo2/revo2.usda`, 압축본은 같은 상위 폴더의 `revo2.zip`입니다.
원본 USD의 내부 폴더·상대 참조와 공유본 내용은 유지했습니다.
