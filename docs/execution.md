# 저장 궤적 실행

정책이 팔 연결 시뮬레이션에서 생성한 **RB3 6축 + Revo2 독립 6축 명령**을 저장합니다.
재생은 저장 명령만 사용하므로 물체 pose estimation·정책 추론·온라인 IK가 필요하지 않습니다.
캔은 기록된 위치·방향에 놓아야 하며, 위치 변화나 미끄러짐을 자동 보정하지 않습니다.

```bash
./run.sh execute inspect
./run.sh execute replay                 # 창 없는 물리 재생
./run.sh execute replay --gui           # 시뮬레이터에서 보기
./run.sh execute dry-run                # 로봇 연결 없는 명령 전송 테스트
./run.sh execute record                 # 새 정책 실행을 별도 폴더에 기록
```

`config/execution.json`에서 데모·팔 배치·기록 파일을 선택합니다.
`record`는 데모의 최신 학습 완료 정책을 학습 당시 속도로 사용합니다. `--checkpoint`로 고정할 수 있습니다.
`replay`와 실물 실행은 저장된 명령 궤적과 그 기록 당시 설정을 유지합니다.
`--recording`, `--checkpoint`, `--arm-config`, `--output`으로 지정할 수도 있습니다.
속도 변경 옵션 없이 기록 시각 그대로 1회 실행합니다. 기본 마지막 자세 유지 시간은 1초입니다.

## 파일과 시간

| 파일 | 내용 |
|---|---|
| `commands.npz` | 명령 시각, 팔·손 12축 rad 목표, 팔 속도 feedforward, 초기 자세, 모델 한계·좌표·checkpoint 식별 정보 |
| `commands.csv` | 같은 관절 명령과 이름, SI 단위 |
| `rollout.npz` | 측정 관절·손목·물체 자세, 접촉·추종 오차, 마지막 유지 구간 |
| `report.json` | 물체 들어올림·유지와 추종 지표 |

명령 시각은 **각 제어 구간 시작 시각**입니다. 30Hz 명령을 4개의 120Hz 물리 step 동안 유지합니다.
측정 관절값과 명령을 혼동하지 않으며, 명령 사이를 새로 평활화하거나 감속하지 않습니다.
기준 궤적 마지막 시각 1.733초의 명령도 한 구간 적용되므로 전체 명령 실행은 1.767초입니다.

## 실물 연결

RB3는 공식 `rbpodo==0.16.14`의 `move_servo_j`, 오른손 Revo2는
`bc-stark-sdk==2.0.3`의 RS485 position API를 사용합니다. 다른 전송 방식은 adapter를 추가합니다.
SDK는 Isaac 환경과 별도 Python 환경에 설치합니다.

```bash
python3 -m venv local/hardware-venv
local/hardware-venv/bin/pip install numpy rbpodo==0.16.14 bc-stark-sdk==2.0.3
cp config/hardware.example.json local/hardware.json

# 설정·단위·관절 한계 검사만 수행: 로봇에 연결하지 않음
DEX_PYTHON=local/hardware-venv/bin/python ./run.sh execute hardware --hardware-config local/hardware.json

# 실제 전송: 현장에서 보정·초기 자세·작업 공간을 확인한 뒤 실행
DEX_PYTHON=local/hardware-venv/bin/python ./run.sh execute hardware --hardware-config local/hardware.json --send
```

`local/hardware.json`에는 실제 IP·포트·baudrate·slave ID, 팔의 부호·영점·물리 한계,
손가락별 **model rad ↔ SDK 0~1000** 측정 보정표와 SDK motor index를 입력합니다.
특히 두 엄지 모터의 순서를 이름만으로 추정하지 않습니다. Servo J gain·filter도 현장 설정을 사용합니다.
`calibration.id`, 책상 배치 fingerprint, 직선 마운트 asset hash는 실제 장착·좌표 보정 결과와 연결합니다.
현재 배치는 베이스 정면 X=0.55m, Y=0이며 상판은 베이스 프레임의 Z=0.02m입니다.

실행기는 초기 자세로 자동 이동하지 않습니다. 시작 전 관절값이 기록 초기 자세와 맞아야 합니다.
관절 제한·feedback freshness·추종 오차·명령 마감 시간을 검사하고 실패 시 전송을 중단합니다.
팔에 stop을 요청하며 손을 자동으로 펴지 않습니다. 모터 활성화·충돌 감지 해제·gain/전류 한계 변경도 하지 않습니다.

실물 Servo J는 위치 명령 API여서 시뮬레이터의 속도 feedforward와 동일한 동역학을 보장하지 않습니다.
Python/RS485 실행 주기는 현장에서 확인해야 하며 지연 시 시간을 늘리거나 명령을 몰아서 보내지 않습니다.
소프트웨어 정지는 물리 비상정지를 대신하지 않습니다.
