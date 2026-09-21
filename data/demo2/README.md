# 2번 데모

실행 선택: `./run.sh floating retarget 2` 또는 `./run.sh arm retarget 2`.

원본 frame ID **15~41**, **27자세**, 현재 재생 **5.2초**.

- `raw/`: DexYCB 원본 annotation·RGB (로컬 보존, Git 제외)
- `poses.npz`: 카메라 좌표의 사람 손 21점·물체 pose
- `retargeted.npz`: 카메라 좌표의 로봇 리타게팅
- `grounded/`: 현재 바닥 좌표의 입력과 좌표 변환 기록
- `policy_reference.npz`: 기존 최신 정책의 정확한 학습 입력
- `manifest.json`: 파일 역할·단위·좌표·프레임·SHA-256

`policy_reference.npz`는 2번 동작의 접촉 자세를 준비한 5.2초 학습 입력입니다.

[두 데모의 설정·실행 방법](../../docs/data.md). 프레임·좌표·궤적 내용은 변경하지 않고 파일만 정리했습니다. 학습 결과와 검증 보고서는 `local/`에 있습니다.
