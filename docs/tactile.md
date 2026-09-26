# Revo2 tactile

정전용량식 Revo2 Touch의 손가락별 출력은 정상력, 접선력, 접선력 방향각입니다.
공식 Modbus 문서 기준 힘은 0–25 N, 통신 단위는 0.01 N/count입니다.
0.01 N/count는 센서 정확도·물리적 분해능을 뜻하지 않습니다.
방향각은 손끝 방향 0°, 시계방향 0–359°이며 `65535`는 무효입니다.
실물의 센서 상태·갱신 번호와 원시 근접값도 보존합니다.

## 시뮬레이션 기록

```bash
./run.sh arm policy 2 --record-tactile --repeat 1
./run.sh floating policy 2 --record-tactile --repeat 1
# 창 없이 측정: --headless 추가
```

기존 정책·물성·제어를 유지하고 120 Hz 물리 스텝 직후 접촉 impulse를 dt로 나누어 N으로 기록합니다.
다섯 패드와 캔·상판의 접촉을 구분하며, 정상 접촉점 힘의 합과 마찰력 벡터를 각각 보관합니다.
패드의 합력은 USD에서 유도한 축으로 투영하여 Revo2 형식의 정상력·접선력·방향각으로 변환합니다.
축은 패드 두께 방향과 distal link→손끝 방향에 근거한 근사이며, 실물 센서 축 보정값은 아닙니다.
범위를 넘는 원본 힘과 포화 플래그를 유지하며, 캔·상판 이외 접촉은 불완전 측정으로 표시합니다.
근접 거리·ADC·노이즈·히스테리시스·실물 대역폭은 추정해서 만들지 않습니다.

실행 폴더의 `tactile/`에 회차별 CSV·NPZ·요약 JSON을 저장합니다.
- `normal_sum_n`: 접촉점의 압축력 합. 방향이 다른 접촉력도 상쇄하지 않습니다.
- `normal_world_n`, `friction_world_n`: 캔/상판별 정상·마찰력 벡터.
- `sensor_normal_n`, `sensor_tangential_n`, `sensor_direction_deg`: 센서 축 근사에서 계산한 값.
- `sensor_registers`: 정상력·접선력·방향각 순서의 공식 통신 형식 근사.
- `sensor_saturated`, `sensor_pair_coverage_valid`: 포화와 접촉 대상 누락 표시.

다섯 손가락의 압축력 합은 물체에 작용하는 순힘이나 손의 최대 파지력 사양과 다릅니다.

## 실물 기록

동작 중에는 [실물 실행·측정 절차](hardware_measurement.md)의 `execute hardware --record-tactile` 또는 ROS bridge를 사용합니다.
아래 독립 기록 명령은 다른 프로세스가 RS485를 사용하지 않을 때만 실행합니다.

프로젝트의 `hardware` extra에 지정된 공식 `bc-stark-sdk==2.0.3`을 사용합니다.
실제 통신 설정을 넣어 실행합니다.

```bash
./run.sh tactile record --port <RS485장치> --slave-id <ID> \
  --baudrate-enum <SDK_Baudrate이름> --seconds 10 --hz 100
```

기록기는 읽기 API만 호출하며 모터·전류·센서 활성화·영점 보정을 변경하지 않습니다.
센서 다섯 개가 이미 활성화되어 있어야 합니다. 센서 실제 갱신 여부는 status의 sequence로 기록합니다.
`--hz`는 PC의 요청 빈도이며 실물 센서의 대역폭을 변경하지 않습니다.
별도 제어기가 같은 RS485 포트를 쓰면 두 연결을 열지 말고, 기존 통신 루프에서
`decode_sdk_sample()`로 SDK 결과를 변환하여 저장합니다.
실물 시간은 PC 요청·응답의 중간 시각이며 시뮬레이션과 비교할 때 동작 시작을 맞춥니다.
비교 전 무부하 영점, 센서 장착 방향과 실물 주기를 확인해야 합니다.

시작 시각 차이를 확인한 뒤 같은 시간 구간의 정상력·접선력 bias/MAE/RMSE를 계산합니다.
갱신되지 않은 실물 패킷, 오류 상태, 시뮬레이션 포화·접촉 누락은 비교에서 제외합니다.

```bash
./run.sh tactile compare --sim <실행폴더>/tactile/episode_0001.npz \
  --real <실물기록폴더>/tactile.csv --offset-s <실물시간-시뮬시간> \
  --output local/reports/tactile_comparison
```

[공식 프로토콜](https://staging.brainco.tech/docs/revolimb-hand/en/revo2/modbus_touch.html) ·
[공식 SDK](https://staging.brainco.tech/docs/revolimb-hand/en/revo2/python_sdk.html)

접촉 기록의 `minimum_contact_separation_m`는 손끝–캔/책상의 최소 접촉 간격(m),
`penetration_depth_m`는 음수 간격의 크기입니다. 접촉점이 없으면 간격은 NaN, 깊이는 0이며 `normal_contact_count`로 구분합니다.
2mm 초과를 `penetration_over_2mm`로 표시하며, JSON 요약은 전체 회차 최대값과 파지 구간 손가락별 값을 담습니다.
힘이 0인 접촉 후보도 기하적 겹침 계산에 포함합니다. 실물 tactile은 이 깊이를 직접 측정하지 않습니다.
