# RGB 데이터 제작

RGB 영상에서 손·물체 궤적을 구성하고 EgoPHI 접촉·힘 추정을 추가해 리타게팅 입력으로 내보냅니다.
`drilling`의 첫 동작은 **접근 → 파지 → 들어 올리기 → 유지**입니다. 드릴은 촬영 중 하나의 rigid object이며 나사·트리거 정보나 depth 영상은 필요하지 않습니다.

## 구조와 준비물

```text
config/tasks/drilling/capture.json     작업·입력 경로·드릴 크기와 좌표
config/dataset/{hamer,egophi}.json      외부 모델·checkpoint·Python
assets/tasks/drilling/drill.obj        실제 크기를 아는 드릴 mesh
data/drilling/demo1/
  raw/                                RGB 영상·프레임·boxes·events
  calibration.json                    카메라·좌표 보정
  poses.npz                           리타게팅용 손 21점·물체 pose
  object_mesh.npz                      object-local metre mesh
  retargeting_input.json               좌표·단위·시간·손 선택
  interaction.npz                     선택적 접촉·힘 추정
  events.json                         수동 동작 구간
  manifest.json                       입력·변환·hash·유효 프레임
local/dataset/drilling/demo1/           추정·build·검수 중간 결과
src/dex_manipulation/dataset/           촬영·보정·추정·검수·내보내기
```

```bash
./run.sh dataset init 1 --task drilling
./run.sh dataset status 1 --task drilling
./run.sh dataset --help
```

번호를 바꾸면 해당 작업의 `demoN`을 사용합니다. 기본 task는 이 데이터 제작 명령에 한해 `drilling`입니다.
원본과 기존 결과를 덮어쓰지 않으며 재처리는 새 `--output` 경로 또는 새 데모를 사용합니다.
원본 영상·annotation, 중간 결과, 외부 코드·checkpoint는 Git에서 제외합니다.

일반 처리에는 NumPy/SciPy와 프로젝트의 `dataset` 의존성(OpenCV·trimesh)이 필요합니다.
동영상 시간 추출에는 `ffprobe`가 필요하며, 직접 기록한 timestamp JSON을 `import --timestamps`로 제공할 수도 있습니다.
Isaac Sim은 시작하지 않습니다. Python은 `DEX_PYTHON=/경로/python ./run.sh dataset ...`으로 지정할 수 있습니다.

HaMeR/EgoPHI 실행에는 각각의 공식 저장소·호환 Python 환경·checkpoint가 필요합니다.
HaMeR의 MANO 모델과 EgoPHI의 모델 의존성도 공식 설치 안내에 따라 별도로 준비하고 이용 조건을 확인합니다.
기본 경로는 `local/external/hamer`, `local/external/egophi`, `local/checkpoints/egophi/`입니다.
`config/dataset/*.json`의 `python`, `repo`, `checkpoint`를 지정하거나 실행 시 `--python`을 사용합니다.
입력 준비·보정·검수와 이미 추정한 annotation의 가져오기는 신경망 환경 없이 사용할 수 있습니다.

## 처리 순서

아래 명령은 영상·mesh·측정값과 모델 환경을 준비한 뒤 사용합니다. `/경로/`는 실제 입력으로 바꿉니다.

```bash
# 영상 가져오기. 실제 presentation timestamp를 보존합니다.
./run.sh dataset import 1 --task drilling --video /경로/drill.mp4

# 카메라로 직접 기록할 때는 import 대신 사용합니다.
./run.sh dataset capture 1 --task drilling --camera 0 --seconds 10

# 보드의 내부 코너 개수와 실제 칸 크기를 지정합니다.
./run.sh dataset calibrate 1 --task drilling \
  --images /경로/calibration_images --board 9 6 --square-m 0.02

# 측정한 marker 설정 파일을 사용합니다.
./run.sh dataset track 1 --task drilling --markers /경로/object_markers.json
./run.sh dataset track 1 --task drilling --markers /경로/table_markers.json

# 생성된 HTML에서 손 영역을 지정하고 boxes.json을 raw/에 저장합니다.
./run.sh dataset annotate 1 --task drilling
./run.sh dataset hands 1 --task drilling --python /경로/hamer/bin/python

./run.sh dataset build 1 --task drilling
./run.sh dataset infer 1 --task drilling --python /경로/egophi/bin/python
./run.sh dataset review 1 --task drilling
./run.sh dataset export 1 --task drilling
```

`annotate`·`review`는 HTML 경로만 출력합니다. 브라우저에서 해당 파일을 열어 RGB와 3D 결과를 확인합니다.
검수 화면은 입력 geometry를 기본 표시하며, 최종 pose 예측·접촉 단계 mesh는 별도로 선택합니다. `mesh 투영`으로 RGB와 보정값의 정합을 확인할 수 있습니다. 원본 시선 방향의 3D 보기는 보정된 RGB 오버레이를 뜻하지 않습니다.
손 영역 편집기의 `boxes.json` 다운로드 위치는 `data/drilling/demo1/raw/boxes.json`으로 맞춥니다.
보이지 않는 손은 빈칸으로 남깁니다. 영역 보간 여부는 annotation에 기록되며 3D 궤적을 자동 보간하지 않습니다.
`build`는 새 결과 폴더를 만들고, 이후 명령은 최근 build를 사용합니다. 다른 결과는 `--build /경로/build_directory`로 선택합니다.
`infer`는 선택 단계이므로 접촉 모델 없이도 `build → review → export`를 사용할 수 있습니다.

영상 import는 영상의 실제 프레임 시간을, 직접 capture는 호스트 수신 시간을 기록합니다. 수신 시간을 센서 노출 시간으로 표시하지 않습니다.
촬영 궤적에는 자동 감속·재표본화·평활화를 적용하지 않으며 로봇 재생 시간 설정과 분리합니다.

## 보정과 물체 추적

- [카메라 보정 예제](../config/dataset/calibration.example.json)는 실제 값으로 채웁니다. `K`, `distortion`, `image_size=[width,height]`가 필수입니다.
- 카메라는 OpenCV 좌표 `x_right_y_down_z_forward`, 책상은 `x_right_y_forward_z_up`, 원점은 책상 표면입니다.
- 고정 카메라는 `camera_motion=fixed`, 측정한 `T_world_camera`, `extrinsics_source`, `world_axes`, `world_origin=table_surface`를 선언합니다.
- 움직이는 카메라는 프레임별 `camera_poses.npz`가 필요합니다. 고정 카메라 변환으로 대신하지 않습니다.
- [물체 marker 예제](../config/dataset/object_markers.example.json)와 [책상 marker 예제](../config/dataset/table_markers.example.json)의 코너 좌표는 각각 object/world frame의 metre 값입니다. ArUco 검출 코너 순서에 맞춰 측정합니다.
- `markers`의 각 항목은 `id`와 네 개의 `[x,y,z]`를 담은 `corners_m`으로 구성합니다. marker ID·크기·부착 위치는 실제 측정값을 사용합니다.
- `track`은 측정한 marker의 RGB PnP를 사용합니다. 재투영 오차가 크거나 평면 pose가 모호한 프레임은 invalid로 저장합니다.
- `capture.json.object`의 `mesh_unit_to_m`, `mesh_to_object`, `measurement_source`를 지정합니다. `mesh_to_object`는 크기 변환 후 적용하는 4×4 SE(3)이며 mesh vertex 순서를 유지합니다.

보정 예제의 `null`은 채워야 할 측정값이며 유효한 기본값이 아닙니다.
단일 삼각형 OBJ/PLY/STL mesh를 사용합니다. 신경망이 드릴 형상·실제 크기·책상 좌표를 자동으로 알아낸다고 가정하지 않습니다.
고정 카메라의 측정 변환이 있으면 책상 marker 추적 명령은 생략할 수 있습니다.

## 외부 annotation 계약

모든 NPZ는 pickle 없이 읽을 수 있어야 합니다. `frame_ids`는 RGB 기록과 정확히 일치하며 임의 index 대응은 하지 않습니다.
`timestamps_s`를 제공하면 RGB 시간과 일치해야 합니다. 생성한 annotation과 손 영역 파일은 `capture_sha256`으로 촬영 출처를 연결합니다.
길이는 metre, 변환은 `T_parent_local`의 4×4 행렬입니다. quaternion 순서 추측이 필요하지 않습니다.

| 파일 | 주요 배열 |
|---|---|
| `hands.npz` | `frame_ids`, `joints_camera_m[T,2,21,3]`, `hand_sides=[left,right]`, `joint_names[21]`, `valid[T,2]` |
| 손 mesh | 같은 파일의 `vertices_left_camera_m[T,778,3]`, `vertices_right_camera_m[T,778,3]`, `faces_left[F,3]`, `faces_right[F,3]` |
| `object_poses.npz` | `frame_ids`, `T_camera_object[T,4,4]`, `valid[T]`, 선택적 `reprojection_error_px[T]` |
| `camera_poses.npz` | `frame_ids`, `T_world_camera[T,4,4]`, `valid[T]` |

각 파일의 `metadata_json`은 JSON 문자열입니다. 공통 필드는 `schema=1`, `length_unit=m`, `camera_axes=x_right_y_down_z_forward`입니다.
손은 `coordinate_frame=camera`, `method`, `metric_source`, `metric_calibrated`를 선언합니다.
손 21점의 이름은 `wrist`와 각 `thumb/index/middle/ring/little`의 `mcp/pip/dip/tip`입니다. 순서는 이름으로 대응시킵니다.
물체는 `measurement_source`를, 카메라는 측정 출처와 책상 frame 정의를 함께 제공합니다.
누락한 손/물체는 `valid=false`이며 0으로 채운 자세를 유효한 관측으로 취급하지 않습니다.

HaMeR의 단안 손 위치는 `geometry_status=monocular_estimate`입니다. 이 상태로 검수·내보내기는 가능하지만 metric 접촉 정답으로 취급하지 않습니다.
metric 정렬 없이 리타게팅에 읽어들이려면 `--geometric-debug`가 필요하며 결과도 geometric debug로 분류합니다.
시험용 보정값이나 손 관절 대용점을 사용할 때는 capture 설정의 `assumptions`에 가정을 기록합니다. 이 입력은 `test_assumptions`로 저장되며 접촉 라벨을 학습용 유효값으로 표시하지 않습니다.
실제 측정한 손 관절 anchor가 있으면 선택적으로 정렬할 수 있습니다. RGB-D는 이 파이프라인의 필수 입력이 아닙니다.

```bash
./run.sh dataset align 1 --task drilling --anchors /경로/anchors.npz
./run.sh dataset build 1 --task drilling \
  --hands local/dataset/drilling/demo1/hands_metric.npz
```

anchor 파일은 같은 손 계약과 `metric_calibrated=true`, 측정 출처를 가지며 `anchor_mask[T,2,21]`로 측정점을 선택합니다.
프레임·손별 최소 3개의 비공선 3D 관절 중심이 필요합니다. 허용 정렬 오차는 `align --max-error-m`로 지정합니다.
EgoPHI의 정답 손 위치 보정은 자동 적용하지 않습니다.

## 접촉 라벨과 내보내기

공식 EgoPHI 모델은 양손 mesh 문맥을 입력으로 받습니다. 기본 `missing_hand_policy=skip`은 한 손이라도 없으면 접촉 추론을 건너뜁니다.
따라서 한손 드릴 영상에서도 손·물체 궤적은 만들 수 있지만 접촉 라벨은 비어 있을 수 있습니다.
`zero_context_debug`는 누락 손을 0 문맥으로 넣는 진단용 옵션이며 해당 예측은 유효 라벨로 인정하지 않습니다.

`interaction.npz`는 vertex별 `contact_*`, `force_*_normalized`, `force_*_direction_camera`와 프레임별 `prediction_available`, `context_complete`, `consistency_pass`, `valid`를 저장합니다.
힘의 단위는 `normalized_model_output`입니다. 실제 드릴에서 측정한 Newton 값이나 마찰력·토크가 아닙니다.
독립 물체 추적 결과는 유지하고 EgoPHI pose·mesh 결과를 별도로 기록해 일관성을 검사합니다.

선택적 `raw/events.json`은 다음 형식입니다. 실제 원본 frame ID를 사용합니다.

```json
[{"phase":"grasp","start_frame_id":26,"end_frame_id":40,"source":"manual"}]
```

허용 phase는 `approach`, `grasp`, `lift`, `hold`, `release`입니다. 파일이 없으면 빈 구간 목록을 저장합니다.
내보낸 `poses.npz`는 기존 `joint_3d`·`pose_y` 계약에 `timestamps_s`, `valid`, table frame metadata를 추가합니다.
`retargeting_input.json`과 `object_mesh.npz`를 함께 사용하며 캔 전용 좌표 보정·형상·접촉 보상을 적용하지 않습니다.
드릴 로봇 재생·학습에는 별도의 [task 환경·설정 등록](tasks.md)이 필요합니다.

[EgoPHI](https://siplab.org/projects/EgoPHI) · [공식 EgoPHI 코드](https://github.com/eth-siplab/EgoPHI) ·
[HaMeR](https://github.com/geopavlakos/hamer) · [방법론·자체 설계 구분](PROVENANCE.md#rgb-데이터-제작)
