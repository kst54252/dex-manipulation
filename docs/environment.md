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


## 캔 형상과 키포인트

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
형상 생성 진입점은 `./run.sh data object`, 설정은 `assets/can/spec.json`입니다.

## 손끝 접촉 물성

다섯 `right_{thumb,index,middle,ring,pinky}_touch_link`의 collision mesh에 고무형 접촉을 적용합니다.
손가락 외피·손바닥·팔·캔·책상은 rigid contact를 사용합니다.

| 패드 설정 | 값 |
|---|---|
| 정적 / 동적 마찰 | 1.2 / 1.0 |
| 접촉 강성 | 10,000 N/m |
| 접촉 감쇠 | 20 N·s/m |
| 마찰 결합 | average |

`contact_materials`로 설정하는 spring/damper 접촉 근사입니다.
`materials.py`가 패드 재질을 바인딩하고 checkpoint 물성 복원 후에도 적용합니다.
패드의 1~2mm 정도 눌림을 허용하기 위한 근사이며, 실제 눌림은 힘·접촉점·동작에 따라 달라집니다.
rest offset은 0으로 유지해 표면에서 저항이 생기게 합니다. 2mm는 강제 관통 상한이 아닙니다.
캔·책상·손 외피의 형상과 rigid material, 마찰·모터 힘 제한은 유지합니다.
실물 고무의 압축 특성을 실측해 맞춘 값은 아닙니다.

```bash
./run.sh floating policy 2
./run.sh arm policy 2
./run.sh floating policy 2 --contact-materials checkpoint
```

기본 재생은 패드 프로필을 적용합니다. `checkpoint` 옵션은 학습 당시 물성을 사용합니다.
새 학습에는 현재 설정을 사용하고, 기존 checkpoint·저장 궤적은 변경하지 않습니다.
기존 정책도 재생할 수 있지만 물성이 달라지므로 새 물성에 맞춘 재학습을 권장합니다.
`train --resume`은 기존 학습 설정을 복원하므로 새 접촉 강성을 자동 적용하지 않습니다.
적용값은 `run_metadata.json`의 `execution.contact_materials`에 저장합니다.
[패드 접근·접촉 학습](train.md) · [구현 출처](PROVENANCE.md)

[손끝 접촉력 측정과 실물 tactile 기록](tactile.md)

`--record-tactile`을 붙이면 120Hz에서 손끝–캔·책상의 접촉 간격과 관통 깊이도 저장합니다.
`penetration_depth_m=max(0,-minimum_contact_separation_m)`이며 2mm 초과를 별도 표시합니다.
이는 보고된 패드 접촉점의 기하적 겹침이고, 손 전체 무관통 검사나 실물 센서 측정값은 아닙니다.
