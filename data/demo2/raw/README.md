# Demo 2 raw input — local only

DexYCB 카메라 `839512060362`에서 선택한 프레임 **15~41**의 입력을 이 폴더에 둡니다.

- `color_000015.jpg`~`color_000041.jpg`: 원본 RGB 27장
- `labels_000015.npz`~`labels_000041.npz`: 원본 annotation 27개

이미지와 NPZ는 Git에서 제외되며 로컬에서는 계속 사용할 수 있습니다.
재배치 시 `../manifest.json`의 SHA-256을 확인합니다. 원본이 없으면 공식 배포처에서
받아야 하며, 카메라 번호만으로 원본 시퀀스를 특정할 수는 없습니다.

[DexYCB 출처·다운로드·라이선스](../../../docs/dataset.md).
