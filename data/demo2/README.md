# 데모2

DexYCB 손·물체 데모입니다. Frame ID 15~41, 27자세, 기준 시간 5.2초입니다.

```bash
./run.sh floating retarget 2
./run.sh arm policy 2 --random-can
```

- `raw/`: 원본 RGB·annotation, Git 제외.
- `poses.npz`, `retargeted.npz`: 카메라 좌표의 사람 입력·로봇 궤적.
- `grounded/`: 바닥 좌표 입력과 변환 기록.
- `policy_reference.npz`: 해당 입력을 사용하는 checkpoint용 궤적.
- `manifest.json`: 파일 역할·좌표·프레임·SHA-256.

[데모 구성](../../docs/data.md) · [데이터 출처](../../docs/dataset.md)
