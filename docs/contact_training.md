# 패드 접근·접촉 학습

기하 리타게팅 궤적에서 다섯 패드가 캔 옆면에 접근하도록 손목·손가락을 최적화합니다.
파지 이후에는 손–물체 상대 자세를 유지하고 접근 구간을 부드럽게 연결합니다.
물체 궤적·시간·사람 키포인트는 유지합니다.

| 보상 | 역할 |
|---|---|
| 패드 표면 거리 | 접촉 전 접근 유도 |
| 가장 먼 패드 거리 | 손가락 전체의 접근 유도 |
| 엄지 대향·다섯 패드 접촉 | PhysX 패드–캔 힘으로 접촉 유지 |
| 접촉력 | 작은 압착 유지, 과도한 힘 억제 |
| 물체 추종 | 궤적 오차에 따라 접촉 보상 조절 |

각 120Hz 물리 substep에서 접촉력을 측정합니다. 파지 직전 손가락 목표에 작은 preload를 점진 적용합니다.

## 입력 생성

프로젝트를 설치한 Python 환경에서 실행합니다. 출력 경로는 새 파일을 지정합니다.

```bash
python scripts/contact.py --source data/demo1/stable/reference.npz \
  --output local/references/contact/demo1.npz --start-frame 25 --blend-start .5 --side-contact
python scripts/contact.py --source data/demo2/grounded/reference.npz \
  --output local/references/contact/demo2.npz --start-frame 26 --blend-start .8 --side-contact
```

FK/FCL·SLSQP로 관절·속도·캔·상판 조건을 계산하며 입력 hash와 생성 설정을 JSON에 저장합니다.

```bash
./run.sh train floating 1 -i 1000
./run.sh train floating 2 -i 1000
```

학습 설정은 `config/policy_demo1_contact.json`, `config/policy_demo2_contact.json`입니다.
`contact.py`는 입력 생성, `policy/grasp_v2.py`는 보상과 preload를 담당합니다.
