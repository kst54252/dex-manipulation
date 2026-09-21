# 캔 형상과 키포인트

사용자 지정 복합 원통입니다. **전체 높이 32mm**, 몸통 지름 **73mm**, 하단 **3mm** 구간의 지름 **76mm**입니다.

| 파일 | 내용 |
|---|---|
| `assets/can/spec.json` | 치수·좌표계·질량·샘플링 설정 |
| `assets/can/can.usda` | 시각 메시·rigid body·두 원통 collider |
| `assets/can/keypoints.usda` | 키포인트 표시 |
| `assets/can/keypoints.csv` | object-local 50점과 고정 index |
| `assets/models/can_mesh.json` | 충돌 형상·좌표변환·asset hash |
| `assets/models/can_points.npz` | 표면점·삼각형·barycentric 좌표 |
| `assets/can/original/` | YCB 캔 원본 보관본 |

object-local +Z가 원통 축입니다. 기하 중심은 `(0,0,-0.0041195)m`이며
밑면 Z=-0.0201195m, 윗면 Z=0.0118805m입니다.
질량은 0.15kg, 관성은 두 원통의 균일 밀도 모델로 계산합니다.

표면 삼각형에서 면적 비례 후보 20,000점을 뽑고 farthest-point sampling으로 50점을 선택합니다.
점과 index는 전체 시퀀스에서 고정하고 물체 pose로 변환합니다. 충돌 계산은 두 원통 전체를 사용합니다.
형상 생성 진입점은 `scripts/object.py`, 설정은 `assets/can/spec.json`입니다.
