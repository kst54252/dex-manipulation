# 데모1

DexYCB `20200709_143747_left`의 사람 데모입니다. Frame ID 12~51, 40자세, 기준 시간 5.2초입니다.

```bash
./run.sh floating retarget 1
./run.sh arm policy 1
```

- `raw/`: 원본 전처리 annotation, Git 제외.
- `poses.npz`, `retargeted.npz`: 카메라 좌표의 사람 입력·로봇 궤적.
- `stable/`: 캔 밑면을 수평으로 맞춘 바닥 좌표 입력.
- `grounded/`, `policy_reference.npz`: 해당 입력을 사용하는 checkpoint용 궤적.
- `manifest.json`: 파일 역할·좌표·프레임·SHA-256.

[데모 구성](../../../docs/data.md) · [데이터 출처](../../../docs/dataset.md)
