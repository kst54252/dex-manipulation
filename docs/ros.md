# RB3-730 + Revo2 통합 제어

팔 6축과 손 독립 6축을 하나의 ROS 2 인터페이스로 조작합니다. 한 장치 담당 프로세스가 명령과 상태 수신을 처리하고, RViz는 **수신한 관절값**으로 팔·손을 함께 표시합니다.

```text
브라우저 조작 패널 / ROS action
  → 12축 제어기 → 가상 서보 / Isaac PhysX / 제조사 VCB / RB3 TCP + Revo2 RS485
                    ↓ 상태
              JointState + 진단 → robot_state_publisher → RViz
```

## 실행

```bash
./run.sh robot virtual                  # 로봇 없이 관절 조작 + RViz
./run.sh robot sim                      # Isaac 물리 시뮬레이션 + RViz
./run.sh robot hardware                 # 실물 관절 상태 수신 + RViz
./run.sh robot hardware --enable-motion --allow-jog
./run.sh robot vcb --enable-motion --allow-jog
```

한 명령으로 브리지·RViz·브라우저 패널(`http://127.0.0.1:8767`)을 시작합니다. 시작 자체로 이동 명령을 보내지 않습니다.
패널에서 목표 각도를 입력하고 **Move to targets**를 누르면 팔과 손이 함께 움직입니다. 조작 입력칸과 실행 중인 명령값·측정값·오차를 구분해 표시합니다.
**Copy measured → targets**는 현재 자세를 입력칸에 복사하며, **Load recording start**는 저장 궤적의 시작 목표만 입력합니다.
**Play recording**은 선택된 저장 궤적을 실행합니다. 시작 자세가 맞지 않으면 거부합니다.
**STOP motion**은 제어 정지 요청이며 비상정지는 실물 로봇의 비상정지 장치를 사용합니다. Ctrl+C는 세션을 정리합니다.

| 모드 | 상태의 출처 | 환경 |
|---|---|---|
| `virtual` | 속도 제한·1차 추종 지연을 가진 가상 관절 | 접촉·중력 계산 없음 |
| `sim` | Isaac PhysX가 계산한 실제 관절 상태 | 기존 직선 마운트·책상·캔·접촉 물성, 물리 120Hz |
| `hardware` | RB3 관절 encoder + Revo2 motor feedback | 기본 읽기 전용, 보정 설정 필요 |
| `vcb` | 제조사 가상 컨트롤러 `jnt_ref` + 가상 손 | 별도 VCB VM 필요; 실물 encoder가 아님 |

`--headless`는 RViz·브라우저 자동 실행을 끕니다. 패널 서버는 유지합니다.
`--no-browser`, `--no-panel`, `--port 8768`, `--seconds 30`도 지원합니다.
설정은 `config/robot.json`, 실행 로그·생성 URDF/STL은 `local/robot/sessions/`에 저장합니다.
저장 궤적은 `config/execution.json`을 사용하며 `--recording local/.../commands.npz`로 선택합니다. 정책을 새로 실행하거나 궤적을 자동 교체하지 않습니다.

## 설치와 실물 연결

Ubuntu 24.04 / ROS 2 Jazzy / Python 3.12 환경입니다. ROS 설치 후:

```bash
sudo apt install ros-jazzy-rviz2 ros-jazzy-robot-state-publisher ros-jazzy-control-msgs \
  ros-jazzy-diagnostic-msgs ros-jazzy-tf2-ros python3-venv
./run.sh setup robot
```

`setup robot`은 `local/robot-venv`에 USD·ROS 연동 보조 라이브러리와 제조사 SDK를 설치합니다.
`run.sh robot`이 이 환경을 선택하며, 없으면 USD·YAML이 설치된 기존 `hardware-venv`도 사용합니다. `sim`은 기존 Isaac Python을 사용합니다.
Isaac 물리는 별도로 설치한 Isaac Sim 6.0.1 환경과 `./run.sh setup sim /Isaac환경/bin/python`이 필요합니다.
ROS 경로는 기본 `/opt/ros/jazzy/setup.bash`이며 `DEX_ROS_SETUP`으로 지정할 수 있습니다.

1. RB3 Ethernet과 Revo2 RS485를 연결합니다.
2. [실물 설정](execution.md)에 따라 `local/hardware.json`에 IP·포트·실측 관절 보정·한계·Servo J 설정을 입력합니다. USD 범위를 SDK 보정표로 대신하지 않습니다.
3. `./run.sh robot hardware`로 관절 방향·각도·마운트·상태를 확인합니다.
4. 실물 이동은 `--enable-motion`으로 허용합니다. 수동 조작은 추가로 `--allow-jog`가 필요합니다. 두 장치가 준비되고 피드백이 최신일 때만 실행합니다.

저장 궤적 실행에는 기존 물리 검증과 보정 일치 검사를 유지합니다. 조작은 관절 공간 보간이며 장애물 회피 경로 계획이 아닙니다.
조작 기본 한계는 팔 10°/s, 손 20°/s, 가속도 40°/s²이고, 실물은 한 번에 관절별 최대 5°입니다.
실측 물리 한계·SDK 변환 한계도 함께 검사합니다. 제조사 프로그램이나 다른 bridge에서 동시에 명령을 보내지 않습니다.
VCB 설치·주소·Simulation mode 설정은 [VCB 안내](vcb.md)를 따릅니다.

## ROS 인터페이스

namespace는 `virtual=/dex_virtual`, `sim=/dex_sim`, `hardware=/dex`, `vcb=/dex_vcb`입니다.

| namespace 뒤의 이름 | 형식 | 내용 |
|---|---|---|
| `/jog` | `control_msgs/action/FollowJointTrajectory` | 12개 이름, 현재 자세와 목표 두 endpoint; quintic 보간·속도/가속도 제한 |
| `/follow_joint_trajectory` | 같은 action | 선택된 기록과 일치하는 궤적·마지막 hold endpoint |
| `/joint_states` | `sensor_msgs/msg/JointState` | 독립 관절 12개, rad |
| `/model_joint_states` | 같은 메시지 | 팔 6개 + 손 전체 11개 |
| `/wrist_pose` | `geometry_msgs/msg/PoseStamped` | 팔 feedback FK, `link0` 기준 |
| `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 준비·동작·오류·상태 나이·장치 정보·피드백 출처 |
| `/stop` | `std_srvs/srv/Trigger` | 현재 action 정지 요청; 정지 완료는 action 결과로 확인 |
| `/robot_description`, `/tf`, `/tf_static` | 표준 ROS 메시지 | 현재 USD에서 생성한 팔·손·작업대 표시 모델과 변환 |

표시 모델은 USD의 양쪽 joint frame·axis·직선 마운트·instance mesh를 반영합니다. 실물 손의 종속 5축은 coupling으로 계산하며 추가 encoder 측정으로 취급하지 않습니다. PhysX 모드에서는 종속 관절도 시뮬레이터 값을 발행하며 mimic 계산으로 덮어쓰지 않습니다.
속도·토크를 측정하지 않은 `JointState` 필드는 비워둡니다. 두 실물 장치의 조회는 병렬이지만 동시 샘플링은 아니며 읽기 구간·RB3 clock을 진단 메시지에 포함합니다.

외부 명령·상태 기본 주기는 30Hz이고 RViz 표시 상한은 60Hz입니다. PhysX 장치 내부 피드백은 120Hz로 받아 제어 구간 사이의 상태 지연을 줄입니다. 조작은 제한을 만족하도록 시간을 연장하며 기록 재생 시간은 변경하지 않습니다.
새 goal은 진행 중인 goal을 덮어쓰지 않습니다. 취소·피드백 중단·추종 한계 초과 시 정지합니다. PhysX 장치는 명령이 0.25초 끊기면 현재 자세를 유지합니다.
`sim`은 현재 물리 설정을 재사용하며 articulation self-collision은 현재 설정대로 꺼져 있습니다. RViz에는 충돌 회피 기능이 없으며 시뮬레이터 조작을 실물 충돌 안전성 검증으로 간주하지 않습니다.

실물 화면은 관절 상태 표시입니다. 물체 pose를 측정하지 않으므로 실제 캔 위치나 접촉을 합성하지 않습니다.
다른 시뮬레이터는 동일한 이름·rad 단위로 `/joint_states`를 구독하거나 종속 관절까지 `/model_joint_states`를 구독합니다.
기존 `./run.sh ros mirror --config local/robot/sessions/<세션>/settings.json`으로 Isaac에 상태를 표시할 수도 있습니다. 이 mirror는 물리를 끈 FK 표시이며 `robot sim` 물리 장치와 구분합니다.

기본 탐색 범위는 localhost입니다. 다른 PC와 연결하려면 `ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`, 같은 `ROS_DOMAIN_ID`, QoS와 호스트 시계 동기화를 사용합니다. 실제 장치 feedback에는 `use_sim_time=false`를 사용합니다.
이 제어기는 표준 ROS 메시지를 사용하는 전용 bridge이며 `ros2_control` hardware plugin이나 MoveIt 범용 컨트롤러가 아닙니다.
