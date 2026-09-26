# 손 FK와 리타게팅

사람 손 21점과 캔 표면 50점으로 interaction mesh를 만들고
로봇 손목 SE(3)·Revo2 독립 관절 6개를 최적화합니다. 물체 pose는 입력 궤적을 사용합니다.

| 단계 | 처리 |
|---|---|
| USD 추출 | joint 연결·양쪽 frame·axis·limits·coupling을 JSON으로 저장 |
| FK | 독립 6관절 → 전체 11관절 → 링크·semantic 키포인트 |
| 물체 점 | object-local 고정 50점을 프레임별 pose로 변환 |
| Interaction mesh | 사람/로봇에 같은 Delaunay connectivity 적용 |
| 목적함수 | Laplacian 변형·시간 smoothness·키포인트 위치·손가락 방향 |
| 제약 | 관절 한계·실제 dt 속도·손과 캔 collision geometry |

FK는 NumPy/SciPy로 실행합니다. 리타게팅은 SLSQP와 FCL을 사용하며 이전 프레임 해로 시작합니다.
좌표는 m/rad, column-vector SE(3), quaternion은 XYZW입니다.
REGRIND와 OmniRetarget의 방법 구분은 [출처 문서](PROVENANCE.md)에 있습니다.

## 실행

기존 데모 재생:

```bash
./run.sh floating retarget 2
./run.sh arm retarget 2
```

새 리타게팅은 프로젝트를 설치한 Python 환경에서 실행합니다.

```bash
python -m pip install -e '.[usd]'
./run.sh retargeting retarget --model-dir assets/models \
  --input data/can_grasping/demo2/poses.npz --metadata config/tasks/can_pick/retargeting_demo2.json \
  --output local/results/retargeting/demo2
./run.sh retargeting view --model-dir assets/models \
  --result local/results/retargeting/demo2 --reference-dir data/can_grasping/demo2/raw
```

`trajectory.npz`에는 손목·관절·키포인트·프레임/전이 mask를,
`report.json`에는 목적함수·제약·추종 오차를 저장합니다.
`comparison.html`은 사람/로봇 동작을 같은 frame ID로 재생합니다.
`./run.sh data ground`로 바닥 좌표를 생성하고 [팔 IK](ik.md)에 연결합니다.

## 접촉점 리타게팅

데모2의 사람 손 21점과 캔 pose를 사용해 두 가지 접촉 방식을 비교한다.
기존 정책·궤적은 변경하지 않으며 출력은 새 `local/` 폴더에 저장한다.

```bash
# 사람 손끝을 캔 표면에 투영한 접촉점을 유지: 기하학적 hard constraint
./run.sh retarget-contact hard --output local/contact_hard

# 동일 접촉점에 가상 힘을 적용하고 제거하며 물리 궤적 최적화
./run.sh retarget-contact virtual --output local/contact_virtual

# 저장한 물리 궤적을 가상 힘 없이 다시 확인 (재최적화 없음)
./run.sh retarget-contact virtual --controls local/contact_virtual/controls.npz --output local/contact_replay

# 손 형상이 달라 사람 접촉점을 만족시키기 어려울 때의 명시적 비교 옵션
./run.sh retarget-contact hard --anchor-mode robot_surface_seed --output local/contact_hard_robot
./run.sh retarget-contact virtual --anchor-mode robot_surface_seed --output local/contact_virtual_robot
```

Isaac Python과 `pip install -e '.[contact]'` 의존성이 필요하다.
Hard 방식의 FK·FCL/SLSQP 계산에는 Isaac Sim 실행이 필요 없다. Virtual 방식은 headless Isaac Sim을 실행하며 RL 학습·GUI·실물 제어를 하지 않는다.

설정은 `config/tasks/can_pick/contact_retargeting.json`이다. 캔 형상과 입력 timestamp를 유지한다.
접촉 구간은 원본 frame ID 26–41, 의미 대응은 thumb/index/middle/ring/little과 각 고무 패드다.

| 출력 | 내용 |
|---|---|
| `contacts.npz` | 고정 object-local 접촉점, semantic 대응, 활성 구간 |
| Hard `trajectory.npz`, `report.json`, `comparison.html` | 손목·관절·키포인트, 프레임/전이 valid mask, 제약 오차와 비교 재생 |
| Virtual `controls.npz` | 시간별 12차원 wrist/finger residual과 단위 scale |
| Virtual `unassisted.npz`, `unassisted.json` | 가상 힘 0 N의 실제 상태·적용 target·패드 힘·오차 |
| Virtual `baseline*`, `grip_seed_only*`, `report.json` | 원래 reference, 초기 굽힘 후보, 단계별 최적화와 최종 비교 |

Hard 접촉은 패드 표면까지의 거리를 제한하며 물리적인 파지력을 보장하지 않는다.
실패 프레임은 `valid=false`로 남는다. Virtual의 물체 추종, 다섯 패드 동시 접촉, coupling 적합성은 별도 지표다.
출력은 자동으로 기존 학습·재생 reference를 대체하지 않는다. [방법론 출처](PROVENANCE.md#접촉점-리타게팅).
