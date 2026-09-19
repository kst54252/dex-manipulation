# FK와 리타게팅

`src/dex_manipulation/fk.py`는 추출한 JSON에서 양쪽 joint frame, axis, limits, coupling과 link-local 키포인트를 계산합니다. 실행 FK는 Isaac Sim 없이 사용할 수 있습니다. `retargeting.py`는 floating 손목 SE(3)와 독립 관절만 최적화합니다. 물체 pose는 고정합니다.

## 준비

저장소 루트에서 README의 Python 환경을 설정합니다. 일반 Python은 `pip install -e '.[usd]'`로 추출/FK/최적화 의존성을 설치합니다. 모델은 이미 `assets/models/`에 있으므로 USD 추출은 모델을 갱신할 때만 실행합니다.

```bash
"$PYTHON" scripts/retargeting.py extract \
  --usd assets/USD/rb3_revo2.usd \
  --keypoints assets/revo2/revo2_keypoints.json \
  --can-mesh assets/can/can.usda \
  --can-collision assets/can/can.usda \
  --output local/results/model_extraction
```

새 추출 결과를 확인한 뒤 사용하는 모델 경로를 선택합니다. 원본 USD는 수정하지 않습니다.

현재 캔은 [사용자 치수의 복합 원통](object.md)입니다. `scripts/object.py`가 전체 높이 32 mm 안에 하단 Ø76×3 mm와 위쪽 Ø73×29 mm를 만들고, 외부 표면 50점을 고정 index로 생성합니다. 원본 캔은 `assets/can/original/`에 있습니다. 리타게팅과 policy/IK에 형상 식별값을 전달하므로 새 형상과 이전 reference를 섞을 수 없습니다.

| 파일 | 내용 |
|---|---|
| `assets/models/revo2.json` | joint graph, 독립 6/가동 11 관절, 키포인트 21개, collision hull |
| `assets/models/can_mesh.npz` | 실제 캔 표면 메시 |
| `assets/models/can_mesh.json` | 캔 collision 형상 |
| `assets/models/can_points.npz` | 전체 프레임에서 유지하는 표면점 50개와 index |
| `data/poses/839512060362.npz` | 원본 frame ID와 사람 손/물체 pose |
| `config/retargeting.json` | 명시적 semantic·좌표·시간 해석 |

단위는 m/rad, column-vector 변환이며 quaternion 저장 순서는 XYZW입니다. 현재 데이터는 카메라 X 오른쪽/Y 아래/Z 전방, 오른손, 캔 object index 0입니다. 원 촬영 fps는 미확인입니다. 사용자가 허용한 재생 간격은 5Hz이며 현재 15~41번의 27개 자세에 5.2초입니다. `config/retargeting.json`의 `frame_range: [15, 41]`는 inclusive ID 범위이며 반복 적용해도 프레임이 더 줄지 않습니다. 원본 42~50번은 활성 데이터에서 제외해 `local/archive/before_tail_trim/`에 보관합니다.

## 실행과 재생

```bash
# 먼저 소수 프레임
"$PYTHON" scripts/retargeting.py retarget \
  --model-dir assets/models --input data/poses/839512060362.npz \
  --metadata config/retargeting.json --max-frames 3 \
  --output local/results/retargeting/smoke

# 현재 전체 27프레임
"$PYTHON" scripts/retargeting.py retarget \
  --model-dir assets/models --input data/poses/839512060362.npz \
  --metadata config/retargeting.json --output local/results/retargeting/839512060362

"$PYTHON" scripts/retargeting.py view \
  --model-dir assets/models --result local/results/retargeting/839512060362 \
  --reference-dir data/raw/839512060362
```

출력 `comparison.html`을 브라우저에서 열면 사람 손/캔과 로봇 손/캔을 같은 frame ID로 비교합니다. `trajectory.npz`에는 손목 pose, 독립/전체 관절, 손/물체 키포인트, frame/transition valid mask가 들어갑니다. `report.json`에는 최적화·제약·추종 오차를 기록합니다.

`valid`는 구현한 기하 제약 검사를 뜻합니다. 사람 동작과 완전히 일치하거나 물리적으로 파지에 성공했다는 판정은 아닙니다. collision 검증을 생략한 결과는 `geometric_debug`이며 valid를 true로 만들지 않습니다.

## 정책 기준 궤적

정책·IK·물리 재생은 `data/grounded/reference.npz`를 읽습니다. `scripts/ground.py`는 원본 `data/poses/839512060362.npz`와 `data/reference/839512060362.npz`에 동일한 강체 변환을 적용하여 바닥 좌표의 `poses.npz`, `reference.npz`, `frame.json`을 만듭니다. 원본과 손가락/시간/손–물체 상대 자세는 보존하며 첫 캔 collision 밑면만 바닥 `z=0`에 맞춥니다. 프레임별 캔 높이를 강제로 고정하지 않습니다.

새 리타게팅 결과는 유효성을 확인한 다음 `scripts/ground.py --reference <새 trajectory.npz>`로 바닥 데이터에 반영하고 `scripts/ik.py solve`로 팔 reference를 다시 만듭니다. 리타게팅 실행만으로 정책 입력을 자동 덮어쓰지 않습니다. `data/grounded/poses.npz`로 직접 리타게팅할 수도 있으며 입력 adapter와 출력에 좌표 변환 기록이 전달됩니다. RGB 비교 뷰어는 표시 사본만 원본 카메라로 역변환합니다.

방법과 자체 선택은 [PROVENANCE.md](PROVENANCE.md)에 있습니다. 개발용 테스트·Isaac 비교·분석은 `local/tools/`와 `local/tests/`에서 실행합니다.

손목 SE(3)와 finger 궤적을 RB3 6축 arm + 독립 finger 6축으로 변환하려면 [RB3 IK](ik.md)를 사용합니다. 카메라–베이스 좌표 변환을 명시해야 합니다.
