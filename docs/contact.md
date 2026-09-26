# 손끝 접촉 물성

다섯 `right_{thumb,index,middle,ring,pinky}_touch_link`의 collision mesh에 고무형 접촉을 적용합니다.
손가락 외피·손바닥·팔·캔·책상은 rigid contact를 사용합니다.

| 패드 설정 | 값 |
|---|---|
| 정적 / 동적 마찰 | 1.2 / 1.0 |
| 접촉 강성 | 20,000 N/m |
| 접촉 감쇠 | 20 N·s/m |
| 마찰 결합 | average |

`contact_materials`로 설정하는 spring/damper 접촉 근사입니다.
`materials.py`가 패드 재질을 바인딩하고 checkpoint 물성 복원 후에도 적용합니다.

```bash
./run.sh floating policy 1
./run.sh arm policy 2
./run.sh floating policy 1 --contact-materials checkpoint
```

기본 재생은 패드 프로필을 적용합니다. `checkpoint` 옵션은 학습 당시 물성을 사용합니다.
적용값은 `run_metadata.json`의 `execution.contact_materials`에 저장합니다.
[패드 접근·접촉 학습](contact_training.md) · [구현 출처](PROVENANCE.md)

[손끝 접촉력 측정과 실물 tactile 기록](tactile.md)
