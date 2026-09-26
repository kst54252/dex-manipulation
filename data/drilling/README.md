# Drilling demos

`demo1/`, `demo2/`, … 아래에 각 시퀀스를 둡니다.
첫 동작은 드릴 접근·파지·들어 올리기·유지입니다. RGB만 사용하며 나사·트리거 동작은 필요하지 않습니다.
`raw/`는 원본 입력, `poses.npz`는 손·물체 시계열, `interaction.npz`는 선택적 접촉·힘 추정입니다.
좌표변환·프레임 시간·입력 출처는 데모별 `manifest.json`에 기록합니다.

드릴의 실제 형상과 pose frame을 연결한 뒤 object keypoint를 생성합니다.
`./run.sh dataset init 1 --task drilling`으로 입력 경로를 준비합니다.
[RGB 데이터 제작](../../docs/dataset_capture.md) · [task 구성](../../docs/tasks.md)
