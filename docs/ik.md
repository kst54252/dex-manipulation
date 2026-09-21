# RB3 FK / IK

floating 손목 궤적을 RB3 팔 6관절로 변환하고 Revo2 독립 손가락 6관절과 합쳐
**12-DoF joint reference**를 생성합니다. FK/IK와 Isaac 실행 코드는 분리되어 있습니다.

USD에서 양쪽 joint frame·axis·limits·flange·손 장착 변환을 읽습니다.
팔 관절 순서는 `base, shoulder, elbow, wrist1, wrist2, wrist3`입니다.

```text
T_base_wrist = T_base_source @ T_source_wrist
T_base_flange = T_base_wrist @ inverse(T_flange_wrist)
```

| 계산 | 방법 |
|---|---|
| Pose IK | 위치·회전을 함께 푸는 bounded damped least squares |
| 연속성 | 이전 해 warm start, 가까운 branch 선택 |
| 특이점 | 정규화 Jacobian SVD와 adaptive damping |
| 관절 한계 | 여유 regularization·hard bounds |
| 이동 제한 | min(20°, USD velocity × dt) |
| 성공 판정 | FK 위치 ≤ 1e-5m, 회전 ≤ 1e-4rad 및 관절·특이점 조건 |

설정은 `config/ik_demo1.json`과 `config/ik.json`입니다.
`alignment`는 데모→책상, `workcell`은 책상→베이스 관계를 지정합니다.

```bash
./run.sh arm retarget 1
./run.sh arm retarget 2
./run.sh arm policy 2
```

모델 추출·궤적 생성 진입점은 `scripts/ik.py extract` / `solve`입니다.
결과 `trajectory.npz`에는 timestamp, q_arm/q_finger, wrist 목표/FK pose,
위치·회전 오차, 특이점·joint-limit 지표, success/transition mask를 저장합니다.
CSV 출력은 관절 이름·시각·12축 rad 목표를 담습니다.

`reference.py`의 `JointReference`가 이름 순서에 맞는 목표를 반환하고
`sim.py`의 Isaac adapter가 articulation target으로 전달합니다.
[정책을 팔에 연결](arm_policy.md) · [환경 좌표](scene.md)
