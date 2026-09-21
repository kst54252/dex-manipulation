# 간단한 학습 실행

저장소에서 `./train.sh`를 실행하면 새 학습/이어 학습, 환경, 데모, 학습 횟수를 메뉴로 선택합니다.
`./run.sh train`도 같은 실행기입니다. Isaac Python과 Python 경로는 재생 실행기와 동일하게 자동 설정합니다.
다른 디렉터리에서는 실행기의 절대 경로를 사용합니다. 옵션의 상대 파일 경로는 저장소 루트 기준입니다.

## 바로 실행

| 환경 | 1번 데모 | 2번 데모 |
|---|---|---|
| 플로팅 | `./train.sh floating 1 --iterations 2000` | `./train.sh floating 2 --iterations 2000` |
| RB3 + Revo2 | `./train.sh arm 1 --iterations 2000` | `./train.sh arm 2 --iterations 2000` |

한 번에 선택한 **한 데모만** 학습합니다. 둘을 섞어 학습하는 옵션은 아닙니다.
두 데모 모두 현재 5.2초 입력입니다. 2번은 `data/demo2/policy_reference.npz`의 접촉 준비 궤적,
1번은 `data/demo1/stable/reference.npz`의 밑면이 수평·바닥 밀착인 궤적입니다.
재생 메뉴의 `policy 1`은 수평 교정 전 입력으로 2000iter 학습한 정책입니다. 새 학습은 메뉴의
checkpoint를 초기값으로 자동 사용하지 않으며, 이어 학습/actor 이관은 명시적으로 선택합니다.

기본 동작:

- 4096환경, RSI·augmentation ON, 기존 상판 보호와 보상 사용.
- `--headless --skip-evaluation`: GUI 없이 학습만 수행하며 종료 후 평가·분석·재생을 실행하지 않음.
- 매 iteration 기존 터미널 표에 보상, 오차, 소요/예상 시간 등을 출력. TensorBoard 로그도 저장.
- 횟수 미지정 시 floating 1000iter, arm 2000iter. `-i`는 `--iterations`, `-n`은 `--num-envs`와 같음.
- 새 결과 폴더에 저장. 기존 checkpoint, 입력 데이터, `config/` 파일은 덮어쓰지 않음.

```bash
./train.sh floating 2 -i 1000 -n 4096
./train.sh arm 1 -i 2000 --output local/results/policy/demo1_arm_run1
./train.sh floating 1 -i 2000 --dry-run
./train.sh --help
```

`--dry-run`은 파일을 생성하거나 Isaac/GPU 학습을 시작하지 않고 선택한 설정·명령을 출력합니다.
`--output`은 아직 존재하지 않는 `local/` 하위 폴더만 허용합니다.
`--save-every 100`으로 저장 주기를 지정하며 마지막 iteration에서는 항상 `policy.pt`를 저장합니다.
TensorBoard가 필요 없으면 `--logger none`을 붙입니다.

## 중력과 팔 설정

새 학습은 기존 중력 단계의 값과 시간 비율을 유지하고, 제어 step 경계만 학습 횟수에 비례해 조정합니다.
현재 recipe에서 1000iter는 750iter 경계, 2000iter는 1500iter 경계, 10000iter는 7500iter 경계부터
전체 중력 9.81 m/s²를 사용합니다. 마지막 25% 구간은 전체 중력입니다. 이는 실제 캡처/시연 시간을
바꾸지 않습니다. 매우 짧은 실행에서 같은 제어 step에 겹치는 단계는 마지막 값을 사용합니다.

팔 학습은 `config/policy_arm.json`의 실제 팔 환경·온라인 IK·PPO 설정을 사용합니다.
2번 선택 시 그 설정의 reference와 좌표 설명을 2번 학습 입력으로, 팔 배치를 `config/ik.json`으로
선택합니다. 1번은 `config/ik_demo1.json`을 사용합니다. 손목/손가락 residual 12개 출력은 같지만
물리 RB3의 IK/추종 오차가 관측과 보상에 반영됩니다. 팔 학습 정책은 `arm`에서 재생합니다.

## 데모1 수평 교정 후 학습

기존 첫 자세의 캔은 11.27° 기울어 테두리만 바닥에 닿았습니다. 새 입력은 캔 축을 +Z로 맞추고
바닥을 Z=0으로 정합니다. 손·물체 전체를 같이 옮기며 이후 바닥 관통 프레임도 공통 Z 보정
(최대 2.495mm)을 적용했습니다. 40자세·5.2초·손 관절·손과 캔의 상대 자세를 유지합니다.
첫 프레임의 실제 초기 선속도/각속도는 기존처럼 0입니다. RSI 중간 프레임 초기화는 유지합니다.

```bash
# 새 학습. 기존 tilted-start 정책의 --resume / --initialize-actor를 붙이지 않음.
./train.sh floating 1 -i 2000 --output local/results/policy/demo1_flat_2000

# 학습이 완료된 후에만 직접 재생
./run.sh floating policy local/results/policy/demo1_flat_2000/policy.pt
./run.sh arm policy local/results/policy/demo1_flat_2000/policy.pt
```

4096환경·RSI ON·증강 ON이며 중력은 1500iter 경계에서 9.81에 도달하고 마지막 500iter는 전체 중력입니다.
GUI와 학습 종료 후 자동 평가/재생은 실행하지 않습니다. 기존 정책과 바뀐 reference의 계약은 다르므로
옛 checkpoint를 새 입력에 덮어씌우거나 단순 이어 학습하지 않습니다. 결과 폴더가 이미 있으면 새 이름을 선택합니다.

## 이어 학습과 actor 이관

```bash
# 저장 시점 이후 1000iter 추가. 환경·데모는 checkpoint 옆 설정에서 읽음.
./train.sh --resume local/results/policy/이전실행/policy.pt --iterations 1000

# 새 팔 학습: 같은 데모/입력/모델 계약의 floating actor만 초기값으로 이관
./train.sh arm 1 -i 2000 --initialize-actor local/results/policy/플로팅실행/policy.pt
```

이어 학습에는 checkpoint 옆의 `config.resolved.json`이 필요합니다. 저장된 설정·중력 단계는
변경하지 않으며 optimizer, RSI/중력 진행 상태, 환경별 물성은 기존 runner가 복원합니다.
`--iterations`는 **추가 횟수**이며 중력 커리큘럼을 다시 시작하거나 새 횟수로 늘리지 않습니다.
저장된 병렬 환경 수를 유지해야 하며 다른 환경/데모를 지정하면 실행 전에 거절합니다.
checkpoint 계약 검증은 기존 엄격한 로더를 그대로 사용합니다.

`--initialize-actor`는 이어 학습과 다릅니다. 새 팔 critic·optimizer·탐색 분산과 새 중력 스케줄을
사용하며, floating actor의 평균 출력 가중치와 관측 정규화 통계만 이관합니다. 실제 입력/모델/관측/
residual scale/상판 설정 호환성은 기존 로더가 검사합니다. `--resume`와 함께 사용할 수 없습니다.

## 결과와 재생

기본 저장 위치는 `local/results/policy/demo번호_환경_횟수_시각/`입니다.
`launch.json`에 명령·선택, `config.input.json`에 실행용 설정, `config.resolved.json`에 runner의
최종 설정, `training.jsonl`과 `tensorboard/`에 학습 지표, `policy.pt`에 최종 정책을 저장합니다.
Ctrl+C로 중단하면 마지막 주기 checkpoint 이후 진행분은 저장되지 않을 수 있습니다.

학습 후 사용자가 직접 다음 명령으로 재생합니다. 등록 정책의 `1`/`2`는 자동 교체하지 않습니다.

```bash
./run.sh floating policy local/results/policy/학습폴더/policy.pt
./run.sh arm policy local/results/policy/팔학습폴더/policy.pt
```

실행기 추가 검증은 명령·입력·설정과 CPU 테스트를 대상으로 했습니다. 이 변경으로 새 물리 학습이나
GUI를 실행하지 않았으며, 학습 명령 생성 자체를 파지 성능 검증으로 해석하지 않습니다.
