# Drilling raw input

드릴을 잡고 들어 유지하는 RGB 영상·원본 annotation을 이 폴더에 둡니다. 데이터는 Git에서 제외됩니다.

- `video.mp4`: 가져올 영상. 파일명은 `--video`로 변경할 수 있습니다.
- `rgb/`: `dataset import` 또는 `capture`가 저장하는 RGB 프레임·`capture.json`.
- `boxes.json`: 검수 화면에서 저장한 좌/우 손 bounding box.
- `events.json`: 선택적 수동 동작 구간.

실제 프레임 시간을 보존하며 RGB-D, 나사·지그 pose, 트리거 상태는 필요하지 않습니다.
카메라 보정은 상위 폴더의 `calibration.json`, 실제 크기의 드릴 mesh는 `assets/tasks/drilling/`에 둡니다.
[명령과 입력 형식](../../../../docs/dataset_capture.md)
