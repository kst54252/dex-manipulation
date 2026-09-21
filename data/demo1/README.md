# 1번 데모

실행 선택: `./run.sh floating retarget 1` 또는 `./run.sh arm retarget 1`.
출처는 원본 프로젝트의 `20200709_143747_left` 사람 데모입니다.

원본 frame ID **12~51**, **40자세**, 현재 재생 **5.2초**.

- `raw/`: DexYCB 기반 원본 annotation NPZ (로컬 보존, Git 제외)
- `poses.npz`: 카메라 좌표의 사람 손 21점·물체 pose
- `retargeted.npz`: 카메라 좌표의 로봇 리타게팅
- `stable/`: 현재 수평·바닥 밀착 좌표의 새 학습/리타게팅 입력과 보정 기록
- `grounded/`: 수평 교정 전 입력과 좌표 기록 (2000iter 정책 호환용)
- `policy_reference.npz`: 과거 1000iter 정책의 정확한 학습 입력 (호환용 보존)
- `manifest.json`: 파일 역할·단위·좌표·프레임·SHA-256

`stable/reference.npz`는 밑면이 수평으로 Z=0에 놓이는 현재 5.2초 입력입니다. 새 학습과 `retarget 1`이
이를 사용합니다. `grounded/reference.npz`는 기본 `policy 1`의 2000iter 학습에 사용됐지만 첫 캔이
11.27° 기울어져 있던 입력입니다. 새 수평 입력에는 재학습이 필요합니다.
`policy_reference.npz`는 **과거 1000iter 정책이 학습한 수정 전
캔 방향·1.3초 입력**으로 호환용으로 보존합니다. manifest의 policy_reference 관련 기록도 이 과거 입력을 설명합니다.

[두 데모의 설정·실행 방법](../../docs/data.md). `stable/`은 원본을 보존한 별도 파생 입력이며 고정 좌표 변환과
최대 2.495mm의 손·캔 공통 바닥 보정을 기록합니다. 학습 결과와 검증 보고서는 `local/`에 있습니다.
