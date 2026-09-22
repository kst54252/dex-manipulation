# Residual RL

정책은 기준 궤적에 손목 위치 3·회전 3·독립 손가락 관절 6개의 residual을 더합니다.
플로팅 손목은 자세 PD로, 손가락은 관절 drive로 제어합니다. PPO는 RSL-RL을 사용합니다.

| 구성 | 동작 |
|---|---|
| 관측 | 손·물체 상태, 기준 궤적, phase, 이전 action; actor 67 / critic 94차원 |
| 보상 | 물체 50점 추종, 선·각속도, 손목 자세, action 변화·크기 |
| RSI | 마지막 프레임을 제외한 기준 제어 프레임을 뽑아 자세·속도 초기화 |
| 증강 | XY·yaw 변환, 관측 noise·delay, 물성·질량·gain·관절 기본값 randomization |
| 중력 | 단계별 커리큘럼; 손 링크 중력 OFF, 캔 중력 적용 |
| 제어 주기 | 물리 120Hz, 정책 30Hz |
| 상판 보호 | 링크별 collision 경계 상자의 여유 계산과 손목 Z 보정 |
| 접촉 학습 | 데모2: 접근·대향·동시 다섯 접촉 보상; 데모1: 접촉 보상 없음 |

상판 보상은 실제 손과 보호 전 목표의 여유를 각각 계산합니다.
기본 명령 여유는 4mm, 속도 선행 시간은 20ms입니다.
패드 접촉 설정은 [접촉 물성](contact.md), 접촉 학습은 [입력과 보상](contact_training.md)에 있습니다.

## 실행

```bash
./run.sh train floating 2 -i 1000
./run.sh floating policy 2
./run.sh arm policy 2
```

플로팅 정책의 팔 재생은 실제 팔·손·캔 상태를 정책 좌표계로 변환하고,
손목 PD 응답을 strict IK로 연결합니다. 팔 환경에서 직접 학습하려면 [팔 정책](arm_policy.md)을 사용합니다.

## 코드

| `src/dex_manipulation/policy/` | 역할 |
|---|---|
| `floating_env.py`, `reference.py` | 플로팅 PhysX 환경·기준 궤적·reset |
| `task.py`, `observations.py` | action·보상·관측 |
| `curriculum.py`, `randomization.py` | RSI·중력·증강 |
| `contact_reward.py`, `grasp_reward.py` | 패드–캔 힘 측정·접근/접촉 보상 |
| `arm_play_env.py`, `arm_train_env.py`, `arm_ik.py` | 팔 재생·학습·IK 변환 |
| `ppo.py`, `trainer.py`, `runner.py` | PPO·공통 학습 루프·실행 구성 |
| `evaluation.py` | 전체 궤적 추종·접촉 평가 |
| `play.py`, `playback.py`, `placement.py` | 반복·배속·랜덤 배치 |
| `tracking.py` | 저장된 정책 궤적의 팔 추종 |

체크포인트에는 입력·모델·학습 설정 계약과 물성 상태가 저장됩니다.
재생의 배속·배치·물성 옵션은 실행 metadata로 분리합니다.
