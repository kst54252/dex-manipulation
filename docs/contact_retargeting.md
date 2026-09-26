# 접촉점 리타게팅

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
