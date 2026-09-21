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
| `data/demo2/poses.npz` | 원본 frame ID와 사람 손/물체 pose |
| `config/retargeting.json` | 명시적 semantic·좌표·시간 해석 |

단위는 m/rad, column-vector 변환이며 quaternion 저장 순서는 XYZW입니다. 현재 데이터는 카메라 X 오른쪽/Y 아래/Z 전방, 오른손, 캔 object index 0입니다. 원 촬영 fps는 미확인입니다. 사용자가 허용한 재생 간격은 5Hz이며 현재 15~41번의 27개 자세에 5.2초입니다. `config/retargeting.json`의 `frame_range: [15, 41]`는 inclusive ID 범위이며 반복 적용해도 프레임이 더 줄지 않습니다. 원본 42~50번은 활성 데이터에서 제외해 `local/archive/before_tail_trim/`에 보관합니다.

## 실행과 재생

```bash
# 먼저 소수 프레임
"$PYTHON" scripts/retargeting.py retarget \
  --model-dir assets/models --input data/demo2/poses.npz \
  --metadata config/retargeting.json --max-frames 3 \
  --output local/results/retargeting/smoke

# 현재 전체 27프레임
"$PYTHON" scripts/retargeting.py retarget \
  --model-dir assets/models --input data/demo2/poses.npz \
  --metadata config/retargeting.json --output local/results/retargeting/839512060362

"$PYTHON" scripts/retargeting.py view \
  --model-dir assets/models --result local/results/retargeting/839512060362 \
  --reference-dir data/demo2/raw
```

출력 `comparison.html`을 브라우저에서 열면 사람 손/캔과 로봇 손/캔을 같은 frame ID로 비교합니다. `trajectory.npz`에는 손목 pose, 독립/전체 관절, 손/물체 키포인트, frame/transition valid mask가 들어갑니다. `report.json`에는 최적화·제약·추종 오차를 기록합니다.

`valid`는 구현한 기하 제약 검사를 뜻합니다. 사람 동작과 완전히 일치하거나 물리적으로 파지에 성공했다는 판정은 아닙니다. collision 검증을 생략한 결과는 `geometric_debug`이며 valid를 true로 만들지 않습니다.

## 정책 기준 궤적

정책·IK·물리 재생은 `data/demo2/grounded/reference.npz`를 읽습니다. `scripts/ground.py`는 원본 `data/demo2/poses.npz`와 `data/demo2/retargeted.npz`에 동일한 강체 변환을 적용하여 바닥 좌표의 `poses.npz`, `reference.npz`, `frame.json`을 만듭니다. 원본과 손가락/시간/손–물체 상대 자세는 보존하며 첫 캔 collision 밑면만 바닥 `z=0`에 맞춥니다. 프레임별 캔 높이를 강제로 고정하지 않습니다.

새 리타게팅 결과는 유효성을 확인한 다음 `scripts/ground.py --reference <새 trajectory.npz>`로 바닥 데이터에 반영하고 `scripts/ik.py solve`로 팔 reference를 다시 만듭니다. 리타게팅 실행만으로 정책 입력을 자동 덮어쓰지 않습니다. `data/demo2/grounded/poses.npz`로 직접 리타게팅할 수도 있으며 입력 adapter와 출력에 좌표 변환 기록이 전달됩니다. RGB 비교 뷰어는 표시 사본만 원본 카메라로 역변환합니다.

방법과 자체 선택은 [PROVENANCE.md](PROVENANCE.md)에 있습니다. 개발용 테스트·Isaac 비교·분석은 `local/tools/`와 `local/tests/`에서 실행합니다.

손목 SE(3)와 finger 궤적을 RB3 6축 arm + 독립 finger 6축으로 변환하려면 [RB3 IK](ik.md)를 사용합니다. 카메라–베이스 좌표 변환을 명시해야 합니다.

## 다른 프로젝트의 사람 데모로 새로 리타게팅

`scripts/demo.py`는 metadata가 포함된 DexYCB 전처리 NPZ에서 사람 MANO21과 물체 pose만
가져옵니다. 기존 로봇 reference나 정책은 입력으로 사용하지 않습니다. 원본 프로젝트의 기본
데모 `20200709_143747_left`는 camera `839512060362`, 원본 ID 12~51의 40프레임입니다.
이전 데이터의 15~41번 crop은 이 데모에 적용하지 않습니다.

```bash
DEMO_RUN=local/results/original_demo_20260920
"$PYTHON" scripts/demo.py \
  --source /home/wanjunkim/ARSL/regrind-revo2/outputs/preprocessed/dexycb/20200709_143747_left/dexycb_right_hand_preprocessed.npz \
  --output "$DEMO_RUN/input"

"$PYTHON" scripts/retargeting.py retarget --model-dir assets/models \
  --input "$DEMO_RUN/input/poses.npz" --metadata "$DEMO_RUN/input/retargeting.json" \
  --output "$DEMO_RUN/retarget"

"$PYTHON" scripts/ground.py \
  --poses "$DEMO_RUN/input/poses.npz" --reference "$DEMO_RUN/retarget/trajectory.npz" \
  --config "$DEMO_RUN/input/retargeting.json" --output "$DEMO_RUN/grounded"

"$PYTHON" scripts/retargeting.py view --model-dir assets/models --result "$DEMO_RUN/retarget"
```

입력 폴더가 이미 있으면 import는 덮어쓰지 않으므로 재실행은 새 `DEMO_RUN` 경로를 사용합니다.
adapter는 `mano_joint_coords_right_mano21`의 명시된 sequential MANO21 순서를 사용하고
semantic 이름으로 우리 모델에 연결합니다. 왼손→오른손 변환은 원본 metadata에 선언된
object-local X 반사를 원본 사람 좌표와 대조합니다. 이미 변환된 오른손에 반사를 다시 적용하지
않습니다. 물체 quaternion의 선언된 WXYZ/XYZW 순서를 확인하며 물체 pose는 반사하지 않습니다.

이 데모의 지면 변환은 명시적 `grounding_mode: camera_negative_y_up`입니다.
카메라 −Y를 world +Z로 사용해 들기 방향을 보존하고, 동일한 강체 변환을 전체 손·물체 시퀀스에
적용합니다. 다른 데모의 기본 `initial_object_plus_z`와 구분합니다. 초기 캔 collision 밑면만
Z=0에 맞추며 카메라–책상 실측 calibration으로 간주하지 않습니다. 사용자 지정 복합 원통과
고정 50개 표면점은 유지합니다.

원본 metadata의 재생 시계 30Hz를 보존해 길이는 1.3초입니다. 촬영 FPS를 별도로 검증한 것은
아닙니다. 학습할 때 별도 policy config의 `reference`를 새 `grounded/reference.npz`로 지정하고
`reference_timing: input_timestamps`를 사용합니다. 기존 기본 dataset·checkpoint는 유지합니다.
왼손 원본 RGB에 오른손 변환 결과를 직접 겹치는 것은 맞지 않으므로 위 명령은 3D 비교를 생성합니다.

### 복합 원통의 위아래 정렬

원본 데모의 object +Z는 camera −Y-up world에서 아래를 향합니다. 따라서 새 캔의 형상을
`mesh_to_object=identity`로 붙이면 Ø76×3mm 부분이 위로 갑니다. 초기 원본 데모 실험
`original_demo_20260920/train_1000`에는 이 오류가 있었으며, 해당 정책은 뒤집힌 캔으로 학습했습니다.

현재 importer는 실제 `CollisionBase`/`CollisionBody`의 반지름·높이·중심으로 위아래를 확인합니다.
필요하면 명시된 기하 중심을 축으로 local X 180도 회전하는 **고정** `mesh_to_object`를 생성합니다.
캔 중심의 원본 경로와 사람 annotation은 보존하며, 캔의 visual/collision·COM·표면점은 모두
같은 object pose로 변환됩니다. 단순 화면 반전이나 매 프레임 자세 강제 보정은 하지 않습니다.

수정 후에는 전체 리타게팅과 grounding을 다시 실행해야 합니다. grounding과 새 정책 학습 전에
두꺼운 base가 body보다 아래인지 검사합니다. 기존 정책의 reference hash/관측 조건과 다르므로
예전 checkpoint를 수정된 학습 결과로 취급하지 않습니다. 수정된 실험 입력·검증·뷰어는
`local/results/original_demo_base_down_20260920/`에 있으며 재학습 전까지 학습 완료로 표시하지 않습니다.

## 두 번째 휴먼데모 재생 시간 (2026-09-21)

두꺼운 밑부분을 아래로 정렬한 원본 폴더 휴먼데모의 현재 재생 시간은
첫 번째 동작과 같은 **5.2초**입니다. 40개 자세의 timestamps를 4배 늘렸으며
손목·손가락·물체·키포인트는 그대로입니다. 기존 1.3초 입력은 보존했습니다.
정리된 입력은 `data/demo1/grounded/reference.npz`이며 `config/policy_original.json`과
`config/ik_original.json`에서 함께 사용합니다. 이전 `local/results/original_demo_base_down_20260920/retimed_5p2s/`
입력 경로는 상대 링크로 유지합니다. 30Hz 제어는 유지하고 157시각으로
보간합니다. 원본의 30Hz metadata는 촬영/재생 출처이며 새로운 동작 속도가 아닙니다.
55cm 위치의 새 IK 79/79와 보간 충돌 검사 1,561표본이 통과했습니다.
재학습은 실행하지 않았고, 기존 checkpoint가 변경된 reference에 맞는 정책이라는
의미가 아닙니다. 상세 기록은 해당 `retimed_5p2s/REPORT.md`에 있습니다.
