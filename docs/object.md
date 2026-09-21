# 캔 형상과 키포인트

현재 캔은 **몸통 지름 73 mm, 전체 높이 32 mm, 하단 3 mm 구간 지름 76 mm**의 복합 원통입니다. 위쪽 몸통 높이는 29 mm이며 전체 높이에 하단 3 mm가 포함됩니다.

| 파일 | 역할 |
|---|---|
| `assets/can/spec.json` | 치수·좌표계·질량 가정·표면 샘플링 설정 |
| `assets/can/can.usda` | 외부 참조 없는 시각 메시 + rigid body + 두 원통 collider |
| `assets/can/keypoints.usda` | `can.usda`를 상대 참조하는 키포인트 표시용 USD |
| `assets/can/keypoints.csv` | 고정 index 0~49와 object-local XYZ(m) |
| `assets/models/can_mesh.npz` | USD에서 추출한 실제 외부 표면 메시 |
| `assets/models/can_mesh.json` | 두 collider, mesh/asset hash와 형상 식별값 |
| `assets/models/can_points.npz` | 50점, index, 소속 triangle과 barycentric 좌표 |
| `assets/can/original/` | 기존 YCB 메시·텍스처·물리 USD·추출 모델 보관본 |

동작에는 `can.usda`를 사용하고, 점을 확인할 때는 `keypoints.usda`를 엽니다. 점 표시용 객체는 물리/학습 환경에 추가하지 않습니다. 결과 이미지는 `local/results/can/keypoints.png`, 비교 재생은 `local/results/retargeting/839512060362/comparison.html`입니다.

원본 object pose frame을 보존합니다. 축은 local +Z이고 기하 중심은 기존 collision cylinder와 같은 `(0, 0, -0.0041195)` m입니다. 새 밑면 local Z는 `-0.0201195` m, 윗면은 `0.0118805` m입니다. 원본 pose·사람 손·RGB는 바꾸지 않았습니다. RGB에는 기존 YCB 캔이 보이며, 기존 메시 정렬 IoU를 새 형상의 검증값으로 사용하지 않습니다.

충돌에는 Ø76×3 mm와 Ø73×29 mm의 정확한 원통 두 개를 사용합니다. 시각 메시에는 하단 턱의 윗면인 고리 면과 상하 캡을 포함한 닫힌 외부 표면만 있습니다. 내부 접합면은 샘플 대상이 아닙니다. 원주 256분할 메시와 정확한 collision 원통의 최대 반경 차이는 약 2.9 µm입니다.

50점은 표면 삼각형의 면적 비례 후보 20,000개에서 farthest-point thinning으로 한 번 생성합니다. seed는 20260918이며 전체 프레임에서 좌표·index가 고정됩니다. 작은 고리 면 등 모든 패치에 최소 한 점을 보장하는 샘플러는 아닙니다. 충돌 검사는 50점이 아닌 두 원통 전체를 사용합니다.

질량은 기존 시뮬레이션의 0.15 kg을 유지했으며 새 캔의 실측값이 아닙니다. 질량 중심과 관성은 두 원통 내부가 균일한 밀도로 채워졌다는 가정으로 계산했습니다. 실제 벽 두께·관성은 미확인입니다. 물성 randomization과 접촉 재질은 기존 설정을 유지합니다.

## 다시 생성

저장소 루트에서 README의 Python 환경으로 실행합니다.

```bash
"$PYTHON" scripts/object.py
"$PYTHON" scripts/retargeting.py retarget \
  --model-dir assets/models --input data/demo2/poses.npz \
  --metadata config/retargeting.json --output local/results/retargeting/839512060362
```

새 결과의 `valid` 전체와 `transition_valid[1:]`가 통과했는지 확인한 뒤 적용합니다.

```bash
cp local/results/retargeting/839512060362/trajectory.npz data/demo2/retargeted.npz
"$PYTHON" scripts/ground.py
"$PYTHON" scripts/ik.py solve
"$PYTHON" scripts/retargeting.py view \
  --model-dir assets/models --result local/results/retargeting/839512060362 \
  --reference-dir data/demo2/raw
"$PYTHON" scripts/physics.py floating --loops 3
"$PYTHON" scripts/physics.py arm --loops 3
```

바닥 변환은 모든 collider의 최소 Z를 검사합니다. 첫 캔 pose 원점은 z=20.1195 mm이고 밑면은 z=0입니다. 기존 −90° 팔 배치·27프레임·5.2초 재생 시간은 유지합니다. RL은 이 27자세만 기존 9초 horizon으로 재배정합니다.

reference·ground·IK·뷰어·RL에 형상 식별값을 전달하고 실제 USD hash도 검사합니다. 이전 캔의 reference/checkpoint를 새 결과로 재사용하지 않습니다. 현재 CSV도 새 IK로 재생성했습니다. 이후 IK를 다시 생성하면 [CSV export](ik.md)도 다시 실행합니다.

검증 로그는 Git 제외 영역 `local/reports/can_resize/`에 있습니다. 기하학적 비관통·IK 통과를 실제 파지 성공으로 해석하지 않습니다. 정책 학습은 실행하지 않았습니다.
