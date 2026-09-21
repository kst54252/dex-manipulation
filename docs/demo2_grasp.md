# 데모2 다섯 손가락 접촉 보상

`config/policy_demo2_grasp.json`은 기하 reference를 그대로 사용하고
**원본 frame 26부터 마지막까지** 다섯 패드–캔 접촉 보상을 적용합니다.
5.2초 기준 궤적에서 파지 구간은 2.2~5.2초입니다.

| 항목 | 역할 |
|---|---|
| 개별 패드 접촉 | 다섯 손가락 각각의 접촉 유도 |
| 동시 다섯 접촉 | 같은 물리 substep의 접촉 유지 |
| 접촉 누락 비용 | 손가락 이탈 억제 |
| 과도한 힘 비용 | 패드당 5N 초과 압착 억제 |

접촉 기준은 0.05N이며 같은 환경의 패드–캔 pair force만 사용합니다.
물체 추종 오차에 따라 긍정 보상을 줄이고 control dt로 적분합니다.
구현은 `policy/grasp.py`, 전용 runner는 `policy/demo2.py`입니다.

## 전용 실행기

```bash
./train_demo2.sh --iterations 2000 --num-envs 4096
./train_demo2.sh --checkpoint local/results/policy/my_grasp_run/policy.pt --iterations 1000
./train_demo2.sh --mode play --checkpoint local/results/policy/my_grasp_run/policy.pt
```

이 보상 전용 checkpoint는 `train_demo2.sh`로 학습·재생합니다.
기본 `run.sh train floating 2`는 [접근 보상과 접촉 궤적을 함께 사용하는 설정](contact_training.md)입니다.
결과는 `local/results/policy/`에 저장하며, 접촉력·동시 접촉률을 학습 로그에 기록합니다.
