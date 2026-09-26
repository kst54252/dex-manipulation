# 다른 PC에서 실행

Ubuntu에서 저장소를 clone한 뒤 실행 환경을 설치합니다. 원본 RGB/annotation과 학습 중간 checkpoint는 필요하지 않습니다.
데모2 2000iter 정책, reference, 팔 IK 궤적, 랜덤 위치 지도,
베이스 정면 55cm의 12축 고정 명령·검증 파일·비교용 tactile 기록은 `runtime/`에 포함됩니다.

```bash
git clone https://github.com/kst54252/dex-manipulation.git
cd dex-manipulation
./run.sh check
```

첫 실행에서 필요한 입력을 `local/`에 복원합니다. 기존 파일은 덮어쓰지 않고,
정책 파일의 원래 수정 시각을 보존해 이후 로컬에서 학습한 최신 정책 선택을 방해하지 않습니다.
`check`는 SHA256 불일치를 보고합니다. Git LFS나 별도 데이터 다운로드는 필요하지 않습니다.

## 시뮬레이션

현재 실행 환경은 Isaac Sim **6.0.1**, Python **3.12**, Torch **2.11.0+cu128**, RSL-RL **5.4.1**입니다.
Isaac Sim과 호환 GPU 드라이버를 먼저 설치합니다. 프로그램이 Isaac Sim 자체를 자동 설치하지는 않습니다.
Isaac 전용 Python에 의존성을 설치하며, 별도 `usd-core`를 덮어 설치하지 않습니다.

```bash
./run.sh setup sim /Isaac환경/bin/python
DEX_PYTHON=/Isaac환경/bin/python ./run.sh floating policy 2
DEX_PYTHON=/Isaac환경/bin/python ./run.sh arm policy 2
DEX_PYTHON=/Isaac환경/bin/python ./run.sh arm policy 2 --random-can
DEX_PYTHON=/Isaac환경/bin/python ./run.sh arm retarget 2
```

Python이 `~/IsaacLab/.venv/bin/python` 또는 저장소 `.venv/bin/python`에 있으면 `DEX_PYTHON`은 생략합니다.
캔 집기 데모 번호는 `2`이며 기본 무한 반복, `--repeat 1`은 1회입니다.
`--dry-run`으로 입력과 실행 명령을 확인하고 `--headless`로 GUI 없이 재생합니다.
재생은 현재 패드 프로필을 적용합니다. 학습 당시 물성은 `--contact-materials checkpoint`로 선택합니다.

## 실물 고정 궤적·촉각 측정

GPU·Isaac Sim 없이 사용할 수 있습니다. 아래 검사는 실물에 명령을 보내지 않습니다.

```bash
./run.sh setup hardware
./run.sh execute inspect
./run.sh execute dry-run
```

설치기는 `local/hardware-venv`와 비어 있는 연결·보정 예제 `local/hardware.json`을 만듭니다.
실제 IP·시리얼 장치·팔 영점·손가락 단위 보정은 사용하는 장비에 맞춰 입력해야 합니다.
`--send`를 지정해야 실물 명령을 보냅니다. [연결부터 실행·측정까지](hardware_measurement.md)
ROS 기능에는 ROS 2와 해당 Python 환경을 별도로 설치합니다. [ROS 연동](ros.md)

## 배포 입력 갱신

동작별 최신 학습이 완료된 원본 PC에서 실행합니다.

```bash
DEX_PYTHON=~/IsaacLab/.venv/bin/python ./run.sh check
PYTHONPATH=src:.deps ~/IsaacLab/.venv/bin/python scripts/package_runtime.py
```

`runtime/manifest.json`은 포함 파일, 원본 SHA256·수정 시각, 이전 저장소 루트와 정책 iteration을 기록합니다.
원본 checkpoint·명령·reference의 내용은 바꾸지 않습니다. 새 작업의 영상·mesh·외부 추정 모델은
각 작업의 입력 절차에 따라 추가합니다. [RGB 데이터 제작](dataset_capture.md)
