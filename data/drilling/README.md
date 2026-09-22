# Drilling demos

`demo1/`, `demo2/`, … 아래에 각 시퀀스를 둡니다.
`raw/`는 원본 입력, `poses.npz`는 손·물체 시계열, `retargeted.npz`는 리타게팅 결과입니다.
좌표변환·프레임 시간·입력 출처는 데모별 `manifest.json`에 기록합니다.

드릴의 실제 형상과 pose frame을 연결한 뒤 object keypoint를 생성합니다.
설정과 등록 방법은 [task 구성](../../docs/tasks.md)을 참조합니다.
