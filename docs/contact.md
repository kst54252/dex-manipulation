# 손끝 접촉 물성

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
[패드 접근·접촉 학습](contact_training.md) · [구현 출처](PROVENANCE.md)

[손끝 접촉력 측정과 실물 tactile 기록](tactile.md)

`--record-tactile`을 붙이면 120Hz에서 손끝–캔·책상의 접촉 간격과 관통 깊이도 저장합니다.
`penetration_depth_m=max(0,-minimum_contact_separation_m)`이며 2mm 초과를 별도 표시합니다.
이는 보고된 패드 접촉점의 기하적 겹침이고, 손 전체 무관통 검사나 실물 센서 측정값은 아닙니다.
