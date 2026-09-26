# ROS 2 통합 제어와 상태 반영

RB3 6축과 Revo2 독립 6축을 하나의 ROS 2 action으로 실행합니다.
별도 Isaac Sim 프로세스는 **수신한 관절 상태**를 표시합니다.

```text
ROS action → 단일 장치 연결 담당 → RB3 Servo J + Revo2 RS485
                               ↓ 측정 feedback
                       JointState + diagnostics
                               ↓
                      Isaac / 다른 ROS 수신기
```

## 로봇 없이 실행

ROS 2 Jazzy가 설치된 환경에서 터미널을 나누어 실행합니다.
`run.sh ros`는 `/opt/ros/jazzy/setup.bash`를 불러옵니다.
다른 ROS 환경은 `DEX_ROS_SETUP=/경로/setup.bash`로 지정합니다.

```bash
# 터미널 1: 모의 장치, 실제 로봇 연결 없음
./run.sh ros bridge

# 터미널 2: 수신한 상태를 Isaac Sim에 표시
./run.sh ros mirror

# 터미널 3: 선택된 저장 궤적을 팔·손에 함께 실행
./run.sh ros send

# 별도 상태 확인
./run.sh ros status --seconds 5
```

모의 장치는 완벽한 위치 추종을 가정합니다. 물리 파지 평가용 환경은 아닙니다.
제조사 Virtual Control Box로 통신을 시험하려면 [VCB 실행](vcb.md)을 사용합니다.
`mirror --headless --seconds 10`은 화면 없이 수신·USD 반영을 실행합니다.
`config/ros.json`에서 namespace·상태 수신 제한 시간·표시 주기를 설정합니다.

## 실물 연결

[실물 설정](execution.md)의 연결 정보·관절 보정표를 준비합니다.
명령 전송을 활성화할 때는 물리 한계·Servo J 설정·기록 검증도 필요합니다.
ROS 프로세스와 Isaac 프로세스는 서로 다른 Python 환경을 사용해도 됩니다.

```bash
python3 -m venv local/hardware-venv
local/hardware-venv/bin/pip install numpy scipy rbpodo==0.16.14 bc-stark-sdk==2.0.3

# 연결 후 관절 상태 수신만 수행
DEX_PYTHON=local/hardware-venv/bin/python ./run.sh ros bridge \
  --backend hardware --hardware-config local/hardware.json

# 기록 궤적의 ROS 실행도 허용하려면 위 프로세스 대신 실행
DEX_PYTHON=local/hardware-venv/bin/python ./run.sh ros bridge \
  --backend hardware --hardware-config local/hardware.json --enable-motion
```

그다음 `./run.sh ros mirror`, `./run.sh ros send`를 사용합니다.
bridge가 두 장치의 연결을 소유합니다. 별도 `execute hardware`나 제조사 제어 프로그램에서
동시에 명령을 보내지 않습니다. 초기 자세로 자동 이동하거나 반복 후 되감지 않습니다.

## ROS 인터페이스

| 이름 | 형식 | 내용 |
|---|---|---|
| `/dex/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | 팔·손 12축 궤적, 진행 feedback·완료·취소 |
| `/dex/joint_states` | `sensor_msgs/msg/JointState` | 독립 관절 12개, rad; 실물은 측정값, mock/VCB는 가상 상태 |
| `/dex/model_joint_states` | `sensor_msgs/msg/JointState` | 팔 6개 + 손 전체 11개; 종속 5개는 coupling 계산값 |
| `/dex/wrist_pose` | `geometry_msgs/msg/PoseStamped` | 수신 팔 관절의 FK 손목 pose, 베이스 `link0` 기준 |
| `/dex/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 준비·동작·오류 상태, feedback 나이, RB3 clock·상태, 손 모터 상태 |
| `/dex/stop` | `std_srvs/srv/Trigger` | 현재 action 취소 및 팔 stop·손 hold 요청 |

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /dex/joint_states --qos-reliability best_effort
ros2 topic hz /dex/joint_states
ros2 service call /dex/stop std_srvs/srv/Trigger '{}'
```

action은 `config/execution.json` 또는 `--recording`으로 선택한 **기존 기록과 일치하는 궤적**만 받습니다.
다른 궤적은 먼저 기록·검증하고 bridge와 send 양쪽에 같은 `--recording`을 지정합니다.
`FollowJointTrajectory` 메시지를 쓰지만 MoveIt용 범용 보간 컨트롤러나 `ros2_control` hardware plugin은 아닙니다.
명령은 원래 30Hz·구간별 유지 방식입니다. 표준 ROS client는 마지막 hold 종료점까지 포함해야 하며
제공된 `send`가 이를 구성합니다. 시간 변경·미래 시작·일부 관절 명령·effort 명령은 거부합니다.
사용자 position tolerance는 기존 한계를 좁힐 수 있으며 나머지 tolerance 필드는 지원하지 않습니다.

## 시간과 시뮬레이터

상태 수신은 기본 30Hz, 표시 목표는 60Hz입니다. 실제 주기는 장치·RS485·네트워크 지연에 달려 있습니다.
ROS 메시지 stamp는 호스트의 읽기 시작 시각이며, 두 장치는 하나의 읽기 구간에서 병렬로 조회합니다.
동시 샘플링을 보장하지 않으므로 `read_window_s`와 RB3 device time을 diagnostics에 함께 보냅니다.
측정하지 않은 속도·토크는 빈 배열로 둡니다. 실물에서는 명령을 측정 상태로 발행하지 않습니다.
mock/VCB 가상 상태의 출처는 diagnostics의 `feedback_source`로 구분합니다.

상태 QoS는 best-effort·최신 1개입니다. 늦거나 순서가 뒤집힌 상태는 버리고,
0.2초 동안 새 상태가 없으면 마지막 자세를 유지합니다. 외삽하지 않습니다.
Isaac의 물리는 이 화면에서 끄며, 원본 USD 대신 메모리 내 표시 계층만 바꿉니다.
캔 pose는 측정하지 않으므로 실제 물체 위치·접촉은 반영하지 않습니다.

다른 PC·시뮬레이터도 같은 joint 이름·rad 단위와 ROS domain/QoS로 구독할 수 있습니다.
호스트 시계는 동기화하고 실제 로봇 피드백에는 `use_sim_time=false`를 사용합니다.
MuJoCo/Gazebo에서는 받은 이름을 해당 모델의 joint에 대응시키는 adapter가 필요합니다.

`bridge --backend hardware --enable-motion --record-tactile`은 action 동안 같은 RS485 연결로 촉각을 기록합니다.
[동기 기록과 그래프 명령](hardware_measurement.md)을 참조합니다.
