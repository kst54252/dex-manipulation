# 패드 접근·접촉 학습

데모2는 다섯 패드가 캔 옆면에 접근하는 기준 궤적과 접촉 보상을 사용합니다.
물체 궤적·시간·사람 키포인트는 유지합니다. 데모1은 기존 리타게팅 입력을 사용하며
손가락 접촉 보상·접근 보상·preload를 적용하지 않습니다. 고무 패드 물성은 두 데모 모두 사용합니다.

| 보상 | 역할 |
|---|---|
| 패드 표면·가장 먼 패드 거리 | 접촉 전 손가락 전체의 접근 유도 |
| 엄지 대향·다섯 패드 접촉 | 같은 환경의 PhysX 패드–캔 힘으로 접촉 유지 |
| 접촉력 | 작은 압착 유지, 과도한 힘 억제 |
| 물체 추종 | 궤적 오차에 따라 접촉 보상 조절 |

원본 frame 26부터 마지막까지 각 120Hz 물리 substep의 접촉력을 측정합니다.
접촉 기준은 0.05N, 과도한 힘 기준은 패드당 5N입니다.
기본 설정은 파지 직전 손가락 목표에 작은 preload를 점진 적용합니다.

## 입력과 설정

프로젝트를 설치한 Python 환경에서 실행하며 출력은 새 파일을 지정합니다.

```bash
python scripts/contact_reference.py --source data/demo2/grounded/reference.npz \
  --output local/references/contact/demo2.npz --start-frame 26 --blend-start .8 --side-contact
./run.sh train floating 2 -i 1000
```

FK/FCL·SLSQP로 관절·속도·캔·상판 조건을 계산하고 입력 hash와 생성 설정을 기록합니다.
`config/policy_demo2_contact.json`은 접근·접촉 보상과 preload를 함께 사용합니다.
기하 궤적을 유지하고 접촉력 보상만 사용하려면 다음 설정을 선택합니다.

```bash
./run.sh train floating 2 --config config/policy_demo2_contact_only.json -i 1000
./run.sh floating policy 2
```

기존 접촉력 전용 checkpoint도 같은 `run.sh` 학습·재생 경로를 사용합니다.
`grasp_reference.py`는 입력 생성, `policy/grasp_reward.py`는 접근 보상·preload,
`policy/contact_reward.py`는 접촉력 보상을 담당합니다.
