# RB3 + Revo2 실물 실행과 촉각 측정

저장된 12축 명령을 한 번 실행하면서 다섯 손가락의 정상력(N), 접선력(N), 방향각(°),
실제 관절값을 같은 PC monotonic 시간축으로 기록합니다. 물체 pose 추정이나 정책 추론은 실행 중 사용하지 않습니다.
기본 입력은 demo2의 2000iter 정책으로 기록한 베이스 정면 55cm 동작입니다.

```text
commands.npz → 단일 제어 프로세스 → RB3 Ethernet / Revo2 RS485
                         ├─ commands.jsonl: 보낸 명령·시각
                         ├─ joint_feedback.csv: 측정 관절·추종 오차
                         └─ tactile.csv: 정상력·접선력·방향각·원시 상태
                                      ↓
                           tactile plot / compare
```

## 1. 장치 연결

1. 현재 직선 마운트로 Revo2를 장착하고 책상·베이스·캔 배치를 확인합니다.
   기록의 캔 중심은 world X=0.55m, Y=0이고 상판은 base Z=0.02m입니다.
   실제 마운트, 패드, 캔 형상·질량과 부착 방향을 맞춥니다. 두꺼운 테두리가 아래입니다.
2. RB3 컨트롤박스와 PC를 Ethernet으로 연결합니다. PC와 컨트롤박스를 같은 subnet의 서로 다른 IP로 설정합니다.
   RB3의 실제 IP는 컨트롤박스에서 확인합니다. 명령은 TCP 5000, 상태는 TCP 5001입니다.
3. Revo2는 해당 모델의 정격 전원·제공 배선도를 사용해 전원을 공급하고, USB–RS485 어댑터로 PC에 연결합니다.
   전원 핀과 A/B/GND 배선을 추정하지 않습니다. 아래 코드는 RS485용이며 CANFD/EtherCAT 연결에는 별도 adapter가 필요합니다.
4. 제조사 도구에서 오른손·통신 속도·slave ID·Normalized 위치 단위를 확인합니다.
   전원 인가 후 손 위치 보정은 빈손에서 완료합니다. 다섯 tactile 채널을 켜고, 필요한 영점 보정도 무부하에서 수행합니다.
   촉각 parameter calibration 직후 10초 데이터는 사용하지 않습니다.
5. RB3 제조사 절차로 실제 운전 모드·초기화·페이로드를 설정합니다. 실행기는 활성화나 모드 변경을 자동 수행하지 않습니다.
   실제 실행 중에는 비상정지에 접근할 수 있어야 하며 작업 공간을 비웁니다.
6. 이 프로그램을 시작하기 전에 다른 RS485 제어기·센서 기록기를 닫습니다. 같은 포트를 두 프로세스가 열지 않습니다.

[RB TCP 통신](https://rainbowrobotics.github.io/rb_cobot_docs/technical_docs/socket_communication) ·
[Revo2 연결 SDK](https://staging.brainco.tech/docs/revolimb-hand/en/revo2/python_sdk.html) ·
[위치·촉각 보정과 데이터 단위](https://staging.brainco.tech/docs/revolimb-hand/en/revo2/modbus_touch.html)

## 2. Python 환경과 설정 파일

저장소 최상위에서 실행합니다. Ubuntu/Python 3.12 기준이며 Isaac용 Python과 분리합니다.
`--system-site-packages`는 같은 PC의 ROS 2 Python 모듈을 사용하기 위한 옵션입니다.

```bash
python3 -m venv --system-site-packages local/hardware-venv
local/hardware-venv/bin/python -m pip install numpy scipy matplotlib rbpodo==0.16.14 bc-stark-sdk==2.0.3
cp -n config/hardware.example.json local/hardware.json
ls -l /dev/serial/by-id/
```

시리얼 접근 권한이 없으면 `sudo usermod -aG dialout "$USER"` 후 로그아웃·로그인합니다.
`/dev/ttyUSB0`보다 `/dev/serial/by-id/...`의 고정 장치 경로를 권장합니다.
`run.sh execute hardware`, `execute probe`, `execute inspect`, `execute dry-run`, `tactile`은 이 venv가 있으면 자동으로 사용합니다.

먼저 `local/hardware.json`의 아래 연결 항목을 채웁니다. 예제의 `null`은 미확인 값입니다.

| 항목 | 넣을 값 |
|---|---|
| `rb3.address` | 컨트롤박스에서 확인한 IP |
| `rb3.command_port`, `rb3.data_port` | 실제 포트; 기본 5000 / 5001 |
| `revo2.port` | RS485 장치 경로 |
| `revo2.baudrate_enum` | 실제 baudrate에 맞는 SDK enum. 예: 실제 460800bps일 때 `Baud460800` |
| `revo2.slave_id` | 장치에서 확인한 ID |

## 3. 움직이지 않고 연결·센서 검사

```bash
./run.sh execute probe --hardware-config local/hardware.json \
  --record-tactile --seconds 5 --output local/results/hardware/probe01

./run.sh tactile plot --real local/results/hardware/probe01/tactile.csv \
  --output local/reports/hardware/probe01
```

`probe`는 RB3의 상태 포트만 열고, 손에는 읽기 API만 사용합니다.
관절 보정표가 없어도 RB3 원시 degree, Revo2 원시 위치·단위·motor state, 촉각을 확인할 수 있습니다.
`raw_feedback.jsonl`과 `probe.json`을 저장하며 정상 동작에서도 센서가 갱신되는지 확인합니다.
무부하 정상력·접선력이 큰 값이면 보정·접촉 여부부터 확인합니다. 측정 코드가 임의로 값을 0으로 빼지 않습니다.

## 4. 관절·좌표 보정과 초기 자세

```bash
./run.sh execute inspect
```

출력의 관절 이름, `initial_q_rad`, `calibration_reference`를 기준으로 실물 설정을 완성합니다.

| 설정 | 확인 방법 |
|---|---|
| `calibration.arm_sign`, `arm_offset_deg` | J1~J6 각 축의 model rad ↔ 실물 degree 부호·영점 확인. `hardware_deg = rad2deg(model_rad) * sign + offset_deg` |
| `calibration.fingers[].sdk_index` | 실제 SDK 모터와 6개 모델 관절 대응. 특히 Thumb Flex/Aux 순서를 확인 |
| `fingers[].rad`, `units` | 같은 여러 자세에서 실측한 모델 각도와 Normalized 0~1000 위치 쌍. rad 오름차순, units 단조; 궤적 전체 범위 포함 |
| `calibration.id` | 사용한 실측 보정의 식별자 |
| `workcell_fingerprint`, `mount_asset_sha256` | `inspect`의 기록 식별자. 실물 배치·장착이 대응됨을 확인한 뒤 기입 |
| `rb3.lower_deg`, `upper_deg`, `velocity_deg_s`, `acceleration_deg_s2` | 해당 RB3의 실제 운전 한계 |
| `rb3.servo` | 현장에서 검증한 Servo J 설정. 이 궤적의 `t1_s`는 1/30초; `t2_s`, `gain`, `alpha`는 임의 지정하지 않음 |
| `revo2.velocity_units_s` | SDK motor index 순서의 실제 운전 속도 한계 |
| `simulation_validation` | 선택한 `commands.npz` 해시와 일치하는 기존 검증 파일 |

USD의 관절 최소·최대만 SDK 0~1000에 비례 대응시키는 것은 실측 보정이 아닙니다.
제조사 도구로 캔 없이 여러 안전한 자세를 만들고 실제 관절각과 raw feedback을 대응시킵니다.
실물에서 사용하는 손 전류·힘 제한, servo 설정, payload도 기록해 두고 실험 간 유지합니다.
[Servo J 매개변수](https://rainbowrobotics.github.io/rb_cobot_docs/technical_docs/ui_script)

로봇 연결 없이 명령값 변환·한계·기록 검증을 검사합니다.

```bash
./run.sh execute hardware --hardware-config local/hardware.json --record-tactile
```

출력의 `rb3_initial_deg`, `revo2_initial_units`가 실물 시작 자세입니다.
제조사 제어기로 주변 간섭을 확인하며 해당 자세로 이동한 뒤 다른 제어 프로그램을 닫습니다.
이 실행기는 초기 자세로 자동 이동하지 않습니다. 시작 오차가 설정 허용치를 넘으면 첫 명령 전에 중단합니다.

현재 저장 궤적의 충돌 검증은 이상적인 coupling을 사용한 표본 검사입니다.
물리 replay에는 최대 약 0.0404rad coupling 차이가 있었고 self-collision이 비활성화되어 있으므로,
이 파일을 실물 전체 경로의 무충돌 보증으로 취급하지 않습니다. 실물 장착·작업 공간 확인이 필요합니다.

## 5. 한 번 실행하면서 기록

```bash
./run.sh execute hardware --hardware-config local/hardware.json \
  --send --record-tactile --tactile-hz 100 \
  --output local/results/hardware/demo2_trial01
```

`--send`가 실제 모터 명령을 활성화합니다. 같은 속도로 1.767초 명령을 적용하고 기본 1초 유지 후 종료합니다.
무한 반복하지 않습니다. 다음 측정은 `trial02`처럼 새 출력 폴더를 사용하고 초기 자세·캔 위치를 다시 맞춥니다.

제어는 저장된 30Hz를 유지합니다. 센서는 명령 사이 여유 시간에만 최대 100Hz로 요청하며,
설정된 I/O timeout 전체가 다음 명령 전까지 들어갈 때만 읽습니다. 부족한 폴링은 건너뛰고 횟수를 기록합니다.
100Hz는 실제 센서 대역폭이나 보장 주파수가 아닙니다. 실측 폴링률·최대 공백·손가락별 갱신 수는 `measurement.json`에 남습니다.
오류 패킷·반복 패킷도 원본 CSV에는 보존하며 그래프·비교에서 구분합니다.
`tactile_usable=false`이면 유효한 센서 표본이 부족하므로 제어 완료와 별개로 측정 성공으로 보지 않습니다.

| 파일 | 내용 |
|---|---|
| `tactile.csv` | 시간별 5손가락 힘·방향각·raw count·근접값·status·갱신 여부 |
| `joint_feedback.csv` | 실제 12축 rad, 그 측정 시 적용 중인 목표와 오차, RB3 device time |
| `commands.jsonl` | 30Hz 예정 시각·전송 시작/완료 시각·명령값 |
| `measurement.json` | 공통 시작 시각, 명령 파일 해시, 센서 펌웨어, 실제 측정률 |
| `report.json` | 완료 또는 중단 이유, 전송 명령 수 |
| `hardware_config.json`, `plan.json` | 실행에 사용한 보정·통신·guard 설정과 입력 |

`Ctrl+C`, 통신 오류, 초기/추종 오차, 명령 deadline 초과 시 팔 stop과 손 hold를 요청하고 기록을 남깁니다.
종료 시 손은 자동으로 펴지 않습니다. 캔을 지지·회수한 뒤 제조사 절차로 손을 펴고 로봇을 대기 자세로 옮깁니다.
소프트웨어 stop은 물리 비상정지를 대체하지 않습니다. 정상 종료 또는 stop/hold 실패는 `report.json`에서 확인합니다.

## 6. 같은 형태의 그래프와 시뮬레이션 비교

실물 값만 그립니다. 센서 오류·미갱신 값은 빈 구간이며, 평활화하지 않습니다.

```bash
./run.sh tactile plot --real local/results/hardware/demo2_trial01/tactile.csv \
  --grasp-start-s 0.7333333333 --output local/reports/hardware/demo2_trial01
```

출력은 `tactile.png`, `tactile.pdf`입니다. 정상력·접선력·방향각을 이전 그래프와 같은 색상으로 표시합니다.
방향각은 손끝 방향 0°, 시계방향이며 0/360°를 선으로 잇지 않습니다.
0.733초 점선은 reference의 파지 구간 표시이며 실물의 실제 접촉 시작을 강제하지 않습니다.

정량 비교에는 **실물에 보낸 고정 명령과 같은 궤적**의 시뮬레이션 기록을 사용합니다.
일반 `arm policy 2` 재실행은 관측에 따라 명령이 달라질 수 있습니다.

```bash
# 선택한 commands.npz 그대로 재생. GUI 없이 실행하고 종료합니다.
./run.sh execute replay --record-tactile --output local/results/hardware/demo2_sim

./run.sh tactile plot \
  --sim local/results/hardware/demo2_sim/tactile/episode_0001.npz \
  --real local/results/hardware/demo2_trial01/tactile.csv \
  --grasp-start-s 0.7333333333 --output local/reports/hardware/demo2_trial01_comparison

./run.sh tactile compare \
  --sim local/results/hardware/demo2_sim/tactile/episode_0001.npz \
  --real local/results/hardware/demo2_trial01/tactile.csv \
  --output local/reports/hardware/demo2_trial01_metrics
```

명령 파일 해시가 같을 때만 예정된 첫 명령을 자동으로 0초 정렬합니다.
통신·센서 내부 지연까지 제거하는 동기화는 아니며, 요청/응답 시각도 함께 확인합니다.
별도 제어기로 측정한 CSV는 동작 시작 이벤트로 측정한 `--offset-s`를 지정합니다.
정의는 `simulation_time = real_time - offset_s`입니다. 힘 모양이 잘 겹치도록 임의로 정렬하지 않습니다.
정상력·접선력의 bias/MAE/RMSE를 계산하며, 방향각의 USD 축은 실물 보정 전 근사입니다.

## ROS 제어기를 사용하는 경우

직접 실행 대신 하나의 bridge가 연결을 소유하도록 합니다.

```bash
# 터미널 1
DEX_PYTHON=local/hardware-venv/bin/python ./run.sh ros bridge \
  --backend hardware --hardware-config local/hardware.json \
  --enable-motion --record-tactile --tactile-hz 100

# 터미널 2: 한 번 실행
./run.sh ros send
```

회차 기록은 action 결과에 출력되는 `local/results/ros/<시각>/`에 저장됩니다.
그래프에는 그 폴더의 `tactile.csv`를 사용합니다. 별도 `tactile record`를 동시에 실행하지 않습니다.
관절 상태·진단·stop service·다른 시뮬레이터 연동은 [ROS 설명](ros.md)을 따릅니다.
ROS bridge의 `Ctrl+C` 또는 `/dex/stop`도 같은 stop/hold 경로를 사용합니다.

## 다른 PC로 옮길 때

실물 고정 궤적 실행·측정·그래프에는 Isaac Sim이나 GPU가 필요하지 않습니다.
ROS 명령을 사용할 때만 ROS 2를 별도로 설치합니다.

- 프로젝트를 복사하되 `local/hardware-venv` 등 Python 가상환경은 제외하고 위 설치 명령으로 다시 만듭니다.
- `config/execution.json`이 가리키는 `commands.npz`와 `local/hardware.json`의 `simulation_validation` 파일을 함께 복사합니다.
  기본 입력은 `local/results/execution/demo2_half_capture_55cm_20260922/` 아래에 있습니다.
- 기존 시뮬레이션 그래프와 비교하려면 해당 `tactile/episode_0001.npz`도 복사합니다.
- Git에서 제외된 `local/`의 입력은 clone만으로 전달되지 않습니다.
- 새 PC에 맞춰 RB3 IP, RS485 장치 경로·권한을 확인합니다. 로봇·마운트·배치가 달라지면 관절·좌표 보정도 다시 확인합니다.
- 새 PC에서 `./run.sh execute inspect`로 입력을 검사한 뒤 읽기 전용 `execute probe`를 진행합니다.

실물 실행은 저장된 정책 checkpoint를 다시 읽지 않습니다. 반면 Isaac의 `execute replay`는
별도 Isaac 환경과 checkpoint가 필요하며 저장 메타데이터의 절대 checkpoint 경로도 확인해야 합니다.
