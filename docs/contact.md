# Revo2 접촉 재질

현재 두 데모의 floating/arm 리타게팅 재생, 새 학습, 기본 정책 재생은 다섯 손끝 패드에만
rubber-like 접촉을 적용합니다. 원본 USD·mesh·질량·관절·제어 gain은 수정하지 않습니다.

## 적용 부위와 값

실제 USD/URDF에서 별도 충돌 mesh를 가진 `right_thumb_touch_link`,
`right_index_touch_link`, `right_middle_touch_link`, `right_ring_touch_link`,
`right_pinky_touch_link`만 선택합니다. 각 부위에 collider가 정확히 하나인지 검사하고,
instance 내부 collider의 physics material binding도 검증합니다. `*_tip_link`는
키포인트용 frame이며, `*_distal_link` 전체를 고무로 바꾸지 않습니다.

| 부위 | 변경 전 | 현재 |
|---|---|---|
| 손끝 패드 5개 | 강체 접촉, 정적/동적 마찰 0.8/0.8, 반발 0 | 정적/동적 마찰 **1.2/1.0**, 접촉 강성 **20,000 N/m**, 접촉 감쇠 **20 N·s/m** |
| 손가락 외피·손바닥·팔 | 기존 강체 접촉 | 그대로 유지 |
| tuna can·책상 | 기존 강체 접촉 | 그대로 유지 |

표의 변경 전 0.8/0.8은 floating 리타게팅 및 학습 환경의 nominal 값입니다.
이전 학습에는 손 마찰 0.7~1.3, 캔 0.5~1.2, 책상 0.6~1.2의 startup randomization이 있었고,
정책 재생은 checkpoint에 저장된 값을 복원했습니다. 정적·동적 계수를 각각 뽑되 동적 계수가
정적 계수보다 크지 않게 제한했습니다. 기존에는 패드 전용 마찰·탄성 구분이 없었습니다.
별도 `scripts/replay.py --mode targets` 팔 궤적 재생은 0.8을 명시하지 않고 USD/PhysX의
기존 재질을 사용해 왔으며, 이번에도 그 경로의 **패드 이외 계수는 바꾸지 않습니다**.

기존 `average` 마찰 결합 규칙을 유지합니다. 따라서 nominal 캔(0.8/0.8)과 패드의
접촉 쌍에서는 정적/동적 마찰이 **1.0/0.9**입니다. 캔–책상 접촉은 기존 강체 접촉입니다.
패드–캔 접촉에서만 패드의 spring/damper가 사용됩니다.

이 값들은 **측정한 Revo2 고무 물성이 아닌 조정 가능한 초기 근사값**입니다. 강성은
탄성계수(Young's modulus)가 아니라 접촉 법선의 force spring 값입니다. 단순 정적
접촉점 모델에서 1 N당 약 0.05 mm 압축에 해당하며 실제 총 접촉력은 접촉점 수·면적·형상에
따라 달라집니다. 고무 mesh가 변형되는 FEM/soft-body 모델은 사용하지 않습니다.
정확한 실물 일치에는 패드 압입 실험과 캔 표면의 미끄럼 실험으로 보정해야 합니다.

## 설정과 실행

`config/policy.json`, `config/policy_original.json`, `config/policy_arm.json`의
`contact_materials`가 전체 프로필을 담습니다. 새 학습 checkpoint에는 이 값도 계약에 저장됩니다.
기존 rigid 부위의 startup randomization은 유지하고, 패드 계수는 해당 프로필 값으로 고정합니다.
재질 tensor를 복원하면 접촉 탄성이 지워질 수 있어 **복원 후 패드 전용 PhysX view에 재적용하고
readback으로 검사**합니다. 강체 collider에는 compliant setter를 호출하지 않습니다.

```bash
./run.sh floating retarget 1
./run.sh arm retarget 1
./run.sh floating policy 1
./run.sh arm policy 1

# 이전 정책이 학습한 물성 그대로 재현할 때만
./run.sh floating policy 1 --contact-materials checkpoint
```

기존 정책의 기본 재생은 새 패드 물성을 사용하므로 **학습 당시 환경과 다릅니다**.
이를 터미널과 `run_metadata.json`의 `execution.contact_materials`에 기록합니다.
이전 checkpoint·학습 설정 파일·contract hash는 변경하지 않습니다. resume 및 기본 evaluate는
학습 설정을 그대로 사용합니다. 새 물성으로 정책을 학습하려면 새 학습을 시작해야 합니다.
이번 물성 변경 작업에서는 학습을 실행하지 않습니다.

## 근거와 검증

방법은 NVIDIA의 [compliant contact 문서](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-materials-for-compliant-contacts)에 따른
암시적 spring/damper입니다. stiffness가 0인 USD material은 rigid contact를 유지합니다.
현재 PhysX tensor의 rigid readback은 +infinity를 반환하므로 이를 soft contact로 오인하지 않습니다.

진단 코드와 결과는 Git에서 제외되는 `local/tools/check_pad_materials.py`,
`local/reports/pad_materials/`에 둡니다. 확인 범위는 실제 collision material binding,
PhysX 마찰/강성/감쇠, startup randomization·checkpoint 복원 후 유지,
짧은 물리 재생의 유한 상태 검사입니다. 실물 물성 보정이나 파지 성공 검증은 아닙니다.

별도 팔 궤적 재생 경로에서는 기존 강체 설정에서도 무효 PhysX 자세가 재현되었습니다.
이 경로를 동일한 TGS solver/반복 설정으로 맞추고, native Newton과 explicit PhysX mimic의
중복을 제거한 뒤 625/625 tick의 재생을 완료했습니다. 재질 외피·질량·drive gain은 유지합니다.
기존 2000iter 정책의 새 물성 재생은 checkpoint 로딩에 성공했으나 물체 오차 종료가 발생했으므로,
이 정책이 새 물성에서 파지에 성공했다고 보지 않습니다.
