# 책상과 로봇 받침대

`config/workcell.json`은 **팔 환경**의 배치입니다.
`scene.py`에서 직육면체 visual/collision을 생성하며 원본 로봇 USD는 수정하지 않습니다.

플로팅 학습·평가·물리 재생은 `config/policy.json`의 `surface`를 사용합니다.
30×30cm 지지면 하나만 생성하고 윗면 z=0, 중심 XY=(0,0), 충돌용 두께 2cm로 둡니다.
실제 reference 첫 캔 위치를 읽어 손·캔을 함께 평행이동하므로 기준 캔 위치가 중앙에 맞습니다.
캔 pose 원점은 현재 z=0.0201195m이고, 형상 밑면이 정확히 z=0에 닿습니다.
학습의 기존 ±5cm XY 증강과 reset noise는 유지하므로 학습 초기 위치는 중앙 주변에서 변할 수 있습니다.
플로팅 환경에는 받침대·책상 다리·별도 하단 바닥을 만들지 않으며 평면 가장자리 밖에는 지지면이 없습니다.

| 구성 | 크기 X×Y×Z (m) | 중심 또는 기준 위치 (m) |
|---|---|---|
| 로봇 받침대 | 0.50×0.50×0.70 | 중심 `(0,0,-0.37)` |
| RB3 장착면 | — | `(0,0,-0.02)`, 회전 없음 |
| 책상 상판 | 0.80×1.60×0.04 | 중심 `(0.65,0,-0.02)`, 윗면 **z=0** |
| 책상 다리 4개 | 0.06×0.06×0.68 | X=0.32/0.98, Y=±0.74, 중심 Z=-0.38 |
| 바닥 | 표시용 2×2×0.04 | 윗면 z=-0.72 |

상판 X 범위는 0.25~1.05m, Y 범위는 -0.8~0.8m입니다.
받침대 앞쪽 X=0.25m에서 책상이 시작합니다. 책상 높이는 바닥에서 72cm,
받침대 높이는 70cm이므로 장착면과 상판의 차이는 2cm입니다.

참고한 것은 `regrind-revo2/config/workcell/rb3_revo2_table.json`의 위 치수·위치·축·장착 회전입니다.
출처 파일 hash는 현재 설정의 `provenance`에 기록했습니다.
컨트롤러·학습·리타게팅·IK·마찰 계수 등은 이 작업에서 해당 저장소로부터 가져오지 않았습니다.
바닥 표시 면적·색상·카메라 시점은 자체 선택입니다.

## 데이터와 시뮬레이터 경계

데이터의 `ground`는 계속 첫 캔 밑면 z=0입니다. 기존 파일을 다시 변환하지 않습니다.
팔은 기존 XY=(0.3,0), yaw=-90°의 world 목표를 유지하고,
로봇 베이스가 -2cm 내려간 만큼 베이스 좌표의 손목 목표 Z가 +2cm 높아집니다.
IK는 이를 반영해 다시 계산합니다. 재생 시 로봇 조립 전체와 정적 root-joint anchor를
함께 배치하고 매 물리 스텝에서 베이스의 실제 위치를 검사합니다.
캔의 world pose에도 같은 `world_from_base`를 사용합니다.

플로팅 환경은 reference의 방향·높이를 유지하고 지지면 중앙에 배치합니다.
관측·보상·저장 rollout의 pose/키포인트는 계속 reference 좌표입니다.
world 좌표로 보려면 metadata의 `task_origins_world_m`를 위치에 더합니다.
이동 속도·회전은 평행이동의 영향을 받지 않습니다. 플로팅 학습 환경 간격은 0.75m입니다.
플로팅 surface를 checkpoint contract에 포함하므로 기존 큰 책상의 checkpoint와 구분합니다.
팔 workcell과 reference 검증은 플로팅 surface 설정의 영향을 받지 않습니다.

## 실행

README의 Python 환경 설정 후 기존 명령을 그대로 사용합니다.

```bash
"$PYTHON" scripts/ik.py solve
"$PYTHON" scripts/physics.py arm --loops 3
"$PYTHON" scripts/physics.py floating --loops 3
```

이번 환경 교체 전 설정·결과는 `local/archive/before_workcell/`, 검증 로그는
`local/reports/workcell/`입니다. 두 폴더는 Git에서 제외됩니다.
플로팅 평면 변경 전 코드는 `local/archive/before_floating_plane/`, 변경 후 검증은
`local/reports/floating_plane/`에 보관합니다. 실행 중인 프로세스에는 환경 변경을 주입하지 않으며 다음 실행부터 적용합니다.
책상·다리·받침대·바닥은 물리 재생에서 collider입니다.
팔 IK에는 환경 collision 제약을 새로 추가하지 않았으므로 IK 성공은
책상과의 무충돌 또는 물리 파지 성공을 뜻하지 않습니다.
