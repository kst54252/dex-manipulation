# Rainbow Virtual Control Box

제조사 VCB 가상 머신으로 RB3의 TCP 명령·응답·상태 수신을 시험합니다.
저장된 팔 6축·손 6축 궤적과 ROS/Isaac 표시 경로를 그대로 사용합니다.

```text
저장 궤적 → ROS action → rbpodo → VCB VM (5000: Servo J, 5001: 상태)
                          └──→ 가상 Revo2 관절 유지
                                       ↓
                           /dex_vcb/joint_states → Isaac 표시
```

## 가상 머신 준비

1. [제조사 안내](https://rainbowrobotics.github.io/rb_cobot_docs/technical_docs/virtual_controlbox)의
   Download에서 `4_Other_Resource/4.6_Window_Simulator_OVA/RBVirtualSimulator.ova`를 받습니다.
2. VirtualBox로 가져오고 **Host-Only Adapter**를 지정합니다. 부팅 완료 후 VM의 실제 IP를 확인합니다.
   문서의 `10.0.2.7`은 예시이며 이 프로젝트는 IP를 자동 추정하지 않습니다.
3. RB Window UI에서 VM에 TCP/IP로 연결하고 RB3-730 모델과 Simulation 모드를 확인합니다.
   Linux UI는 [별도 배포](https://rainbowrobotics.github.io/rb_cobot_docs/technical_docs/use_linux_ui)입니다.
   명령을 보내는 동안 UI에서 동작 명령을 동시에 실행하지 않습니다.

VM은 이 PC 또는 네트워크로 연결되는 다른 PC에서 실행할 수 있습니다.
VCB는 Simulation 모드 전용이며, 포트 5000/5001에 접근할 수 있어야 합니다.
이 도구는 VM 설치·네트워크 설정·로봇 모드 변경을 자동 수행하지 않습니다.

## 연결 설정

```bash
python3 -m venv --system-site-packages local/vcb-venv
local/vcb-venv/bin/pip install numpy scipy rbpodo==0.16.14

./run.sh vcb init --address <VCB_VM_IP>
./run.sh vcb probe
```

`run.sh vcb`는 `local/vcb-venv`를 자동 선택합니다. `mirror`는 Isaac Python을 사용합니다.
`probe`는 데이터 포트만 열어 모드·준비 상태·관절 degree·통신 지연을 출력합니다.
생성한 `local/vcb.json`에서 `mapping.arm_sign`과 `mapping.arm_offset_deg`를 채웁니다.
관절 순서는 `base, shoulder, elbow, wrist1, wrist2, wrist3`입니다.

```text
VCB joint degrees = degrees(USD joint radians) × arm_sign + arm_offset_deg
```

UI의 영점 자세와 각 축의 양의 회전 방향을 USD 모델과 비교해 입력합니다.
영점·방향이 모두 같다고 확인한 경우에만 각각 `[1,1,1,1,1,1]`, `[0,0,0,0,0,0]`을 씁니다.
이 보정값을 실물의 검증된 보정값으로 취급하지 않습니다.

`rb3.servo`의 `t2=0.1, gain=1, alpha=0.5`는 이 프로젝트의 **가상 통신 시험용 시작값**입니다.
제조사 권장값이나 실물 튜닝 결과가 아닙니다. `t1`은 기록의 명령 간격과 같아야 합니다.

## 저장 궤적 실행

```bash
# 기록과 VCB 초기 관절각 확인; 연결하지 않음
./run.sh vcb plan

# VM 팔을 기록 초기 자세로 이동 (10 deg/s); bridge 시작 전에 실행
./run.sh vcb prepare

# 터미널 1: VCB + 가상 손의 ROS bridge
./run.sh vcb bridge

# 터미널 2: 받은 관절 상태를 Isaac에 표시
./run.sh vcb mirror

# 터미널 3: 저장된 팔·손 궤적 1회 실행
./run.sh vcb send
```

관찰만 하려면 `bridge --read-only`, 텍스트 상태는 `status --seconds 5`를 사용합니다.
기본 기록은 `config/execution.json`에서 선택하며 `--recording`으로 바꿀 때는
`plan/prepare/bridge/send`에 같은 파일을 지정합니다. 기록 속도를 자동 변경하거나 반복하지 않습니다.
재실행 전에는 bridge를 종료하고 `prepare`로 초기 자세를 맞춥니다.
읽기·초기화 결과는 `local/results/vcb/`, action 기록은 `local/results/ros/`에 저장합니다.

VCB의 ROS namespace는 `/dex_vcb`로 실물의 `/dex`와 분리합니다.
다른 ROS 수신기는 `/dex_vcb/joint_states`를 rad 단위로 구독합니다.
전체 인터페이스와 QoS는 [ROS 문서](ros.md)를 따릅니다.

## 피드백의 의미

| 항목 | VCB 시험 |
|---|---|
| 팔 | VCB의 `jnt_ref`: 가상 컨트롤러 관절값 |
| 손 | 마지막 적용된 6축 명령을 유지하는 가상 상태; RS485 연결 없음 |
| Isaac 화면 | 수신 상태의 FK 표시; 접촉·파지 물리 평가 없음 |
| 관절 오차 | 저장 명령과 VCB reference의 차이; 실제 encoder 추종 오차 아님 |

feedback 출처는 ROS diagnostics와 action 로그에 기록합니다.
기존 로그의 `q_measured_rad` 필드는 호환성을 위해 유지하며 VCB에서는 위 가상 상태를 의미합니다.
Simulation이 아닌 모드, 오래된 상태, 정지된 device clock, fault, 초기 자세 불일치,
명령 ACK 실패·지연이 발생하면 실행을 중단합니다. VCB에는 물리 모터 활성화를 요구하지 않습니다.
설정된 주소가 실제 VM인지 확인해야 하며, Simulation 비트만으로 VM과 실물 컨트롤 박스를 구별할 수는 없습니다.
