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
./run.sh floating retarget 1
./run.sh arm retarget 2
```

새 리타게팅은 프로젝트를 설치한 Python 환경에서 실행합니다.

```bash
python -m pip install -e '.[usd]'
python scripts/retargeting.py retarget --model-dir assets/models \
  --input data/can_grasping/demo2/poses.npz --metadata config/tasks/can_pick/retargeting_demo2.json \
  --output local/results/retargeting/demo2
python scripts/retargeting.py view --model-dir assets/models \
  --result local/results/retargeting/demo2 --reference-dir data/can_grasping/demo2/raw
```

`trajectory.npz`에는 손목·관절·키포인트·프레임/전이 mask를,
`report.json`에는 목적함수·제약·추종 오차를 저장합니다.
`comparison.html`은 사람/로봇 동작을 같은 frame ID로 재생합니다.
`scripts/ground.py`로 바닥 좌표를 생성하고 [팔 IK](ik.md)에 연결합니다.
