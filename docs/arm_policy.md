# 팔 환경의 정책

손목·손가락 residual을 RB3 + Revo2 고정 베이스 articulation에서 실행합니다.
손목 목표는 온라인 IK로 팔 관절 명령으로 변환합니다.

| 항목 | 플로팅 정책을 팔에 연결 | 팔 환경에서 학습 |
|---|---|---|
| 출력 | 손목 6 + 손가락 6 residual | 동일 |
| Actor / critic | 67 / 94차원 | 87 / 114차원 |
| 손목 제어 | 가상 PD 응답 → CPU IK | 상판 보호 → batched GPU IK |
| 팔 상태 관측 | 손목·손의 실제 상태 | arm q·qdot, IK 잔차·성공·실패 횟수 추가 |
| 팔 관련 보상 | 기본 손·물체 보상 | IK·추종·특이점·관절 한계 비용 추가 |

IK는 이전 해를 시작점으로 삼고 실제 dt의 관절 속도·step 제한을 적용합니다.
위치·회전을 함께 맞춘 뒤 FK 오차와 Jacobian을 검사합니다.
실패 시 직전 관절 목표를 유지하며, 팔 학습에서는 연속 실패 횟수로 종료합니다.

```bash
./run.sh arm policy 2
./run.sh train arm 2 -i 2000 -n 4096
./run.sh arm policy local/results/policy/my_arm_run/policy.pt
```

`config/policy_arm.json`은 학습 환경·보상, `config/ik_demo1.json`과 `config/ik.json`은
각 데모의 모델·장착·좌표변환·IK 조건을 지정합니다.
팔은 joint position/velocity target, 손가락은 USD coupling으로 확장한 target을 받습니다.

`scripts/tracking.py plan`은 저장된 플로팅 정책 궤적을 12-DoF 관절 궤적으로 변환하고,
`replay`는 그 관절 목표를 반복하며 목표/실제 손목·관절 오차를 기록합니다.
[IK와 출력 형식](ik.md) · [학습 옵션](train.md) · [재생 옵션](run.md)
