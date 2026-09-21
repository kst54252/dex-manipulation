# 시뮬레이션 환경

상판 윗면을 **Z=0**, 길이 단위를 m로 사용합니다. 캔은 자유 강체이며 초기화 시 밑면을 상판에 맞춥니다.

| 구성 | 크기 | 배치 |
|---|---|---|
| 플로팅 지지면 | 30×30cm, 두께 2cm | 중앙 XY=(0,0) |
| 팔 환경 상판 | 80×160cm, 두께 4cm | 중앙 XY=(0.65,0)m |
| 로봇 받침대 | 50×50×70cm | 중앙 (0,0,-0.37)m |
| 로봇 장착면 | — | Z=-0.02m |
| 팔 환경 바닥 | 2×2m | 윗면 Z=-0.72m |

플로팅은 정책 설정의 `surface`, 팔 환경은 `config/workcell.json`에서 지정합니다.
`scene.py`가 visual/collision 형상과 로봇 장착 변환을 생성합니다.

팔의 기본 캔 시작 위치는 베이스 중심에서 앞쪽 55cm인 XY=(0.55,0)m입니다.
데모별 `alignment`가 손·물체 전체의 회전과 이동을 정의합니다.
`base_from_source = inverse(world_from_base) @ world_from_source`를 IK에 전달하며
정책 관측은 역변환으로 데모 좌표에 맞춥니다.

```bash
./run.sh floating policy 2
./run.sh arm policy 2
./run.sh arm policy 2 --random-can
```

[물성](contact.md) · [캔 형상](object.md) · [랜덤 배치](run.md#데모2-랜덤-캔-배치)
