# dex-manipulation

사람 손·물체 데모를 Revo2 동작으로 리타게팅하고, RB3-730 팔 IK와 residual RL을 연결하는 Isaac Sim 프로젝트입니다.

```text
DexYCB 손 21점 + 물체 6D pose
  → 손목·손가락 리타게팅
  → 플로팅 Revo2 / RB3 + Revo2
  → 물리 재생·정책 학습
```

## 실행

Isaac Sim과 프로젝트 의존성이 설치된 Python 환경에서 사용합니다.
`run.sh`가 `~/IsaacLab/.venv` 또는 저장소의 `.venv`를 찾고 Python 경로를 설정합니다.
다른 환경은 `DEX_PYTHON=/경로/python ./run.sh`로 지정합니다.

```bash
./run.sh                      # 환경·데모·정책 선택 메뉴
./run.sh floating retarget 1   # 플로팅 손 · 리타게팅
./run.sh arm retarget 1        # 로봇팔 연결 · 리타게팅
./run.sh floating policy 1     # 플로팅 손 · 학습 정책
./run.sh arm policy 1          # 로봇팔 연결 · 학습 정책
```

마지막 번호는 **데모1 / 데모2** 선택입니다. 정책 번호와 checkpoint 경로는 `config/play.json`에서 관리합니다.
체크포인트·접촉 학습 입력·IK 지도는 로컬 `local/`에 보관합니다.

| 옵션 | 동작 |
|---|---|
| `--repeat 3` | 3회 재생. 기본은 무한 반복, Ctrl+C 또는 창 닫기로 종료 |
| `--speed 1` | 1배속. 기본은 플로팅·팔 정책 2배속, 팔 리타게팅 1배속 |
| `--headless` | 창 없이 실행 |
| `--dry-run` | 실행할 입력과 명령 확인 |

```bash
./run.sh arm policy 2 --random-can             # 매 반복 IK 영역에서 캔 위치 변경
./run.sh floating policy local/results/policy/my_run/policy.pt
```

## 학습

```bash
./run.sh train                                # 학습 메뉴
./run.sh train floating 1 -i 1000 -n 4096
./run.sh train floating 2 -i 1000 -n 4096
./run.sh train arm 2 -i 2000 -n 4096
```

학습은 창 없이 실행하며 결과를 `local/results/policy/`에 저장합니다.
`train.sh`도 같은 학습 실행기입니다. [재생 옵션](docs/run.md) · [이어 학습·설정](docs/train.md)

## 프로젝트 구성

| 경로 | 내용 |
|---|---|
| `assets/` | RB3·Revo2 USD, 캔 형상, 키포인트, 추출 모델 |
| `data/demo1/`, `data/demo2/` | 사람 손·물체 데모와 리타게팅 궤적 |
| `config/` | 데모·환경·IK·정책 설정 |
| `src/dex_manipulation/` | FK, IK, 리타게팅, 시뮬레이션, 정책 패키지 |
| `scripts/` | 데이터 처리·모델 추출·학습 진입점 |
| `docs/` | 기능과 설정 설명 |
| `local/` | Git 제외: 실행 결과·체크포인트·테스트·분석·공유 파일 |

[데모 구성](docs/data.md) · [리타게팅](docs/retargeting.md) · [IK](docs/ik.md) ·
[정책](docs/policy.md) · [환경](docs/scene.md) · [캔](docs/object.md) · [접촉 물성](docs/contact.md)

사람 손·물체 데이터는 **[DexYCB](https://dex-ycb.github.io/)**를 사용합니다.
원본 이미지·annotation은 로컬에 보관합니다. [데이터 출처](docs/dataset.md) · [방법론 출처](docs/PROVENANCE.md)
