# 1번·2번 데모

사용자 요청으로 번호를 서로 바꿨습니다. **1번**은 가져온 40자세 동작(`data/demo1/`, 이전 2번),
**2번**은 기존 27자세 동작(`data/demo2/`, 이전 1번)입니다. `data/`에는 이 두 폴더만 둡니다.
학습 로그·정책 가중치·테스트 결과는 계속 Git 제외 `local/`에 둡니다.

두 데모는 **DexYCB 기반 입력**입니다. `raw/`의 이미지·NPZ는 로컬에 보존하며 Git에서 제외합니다.
원본 출처·다운로드·논문 인용·라이선스와 파생 처리 구분은 [dataset.md](dataset.md)에 있습니다.

| 항목 | 1번 — `data/demo1/` | 2번 — `data/demo2/` |
|---|---|---|
| 출처 | 원본 프로젝트의 `20200709_143747_left` | 현재 프로젝트의 기존 데모 |
| 원본 frame ID | 12~51, 40개 | 15~41, 27개 |
| 현재 기하 궤적 길이 | 5.2초 | 5.2초 |
| 원본 입력 | `raw/human_demo.npz` | `raw/`의 RGB·label |
| 리타게팅 설정 | `config/retargeting_original.json` | `config/retargeting.json` |
| 현재 바닥 입력 | `stable/reference.npz` (수평 교정) | `grounded/reference.npz` |
| 팔 IK 설정 | `config/ik_demo1.json` | `config/ik.json` |
| 새 학습/플로팅 설정 | `config/policy_original.json` | `config/policy.json` |
| 재생 메뉴에 등록된 정책의 입력 | `grounded/reference.npz`, **수평 교정 전·2000iter** | `policy_reference.npz`, 5.2초·1000iter |

공통 구조:

```text
data/
  demo1/                   1번 데모 (가져온 40자세)
    raw/                   원본 입력
    poses.npz              사람 손 21점·물체 pose, 카메라 좌표
    retargeted.npz         로봇 손 리타게팅, 카메라 좌표
    grounded/
      poses.npz            바닥 좌표의 사람 손·물체
      reference.npz        이전 바닥 좌표의 로봇 손·물체 궤적 (과거 정책 호환)
      frame.json           좌표 변환·시간 출처
    stable/                현재 수평·바닥 밀착 입력: poses.npz / reference.npz / frame.json
    policy_reference.npz   과거 1.3초 정책의 학습 입력 (호환용 보존)
    manifest.json          파일 역할·frame·hash
    README.md
  demo2/                   2번 데모 (기존 27자세); grounded/가 현재 입력, stable/ 없음
    policy_reference.json  2번 동작의 접촉 reference 준비 기록 (추가 파일)
```

NPZ·원본 RGB·label·변환 기록은 다시 계산하거나 덮어쓰지 않고 바이트 그대로 옮겼습니다.
원본 프로젝트의 전처리 NPZ도 실제 원본 파일의 SHA-256을 확인해 복사했습니다. 리타게팅에는 그 파일의 사람 손·물체 annotation만 사용하며 외부 로봇 궤적이나 정책을 가져오지 않습니다.

두 데모의 `grounded/reference.npz`는 두꺼운 캔 밑부분이 아래를 향하는 현재 5.2초 기하 입력입니다.
1번 동작의 과거 1000iter 정책은 수정 전 1.3초 입력으로 학습됐으므로 `policy_reference.npz`를 호환용으로
보존합니다. 현재 `policy 1`은 밑면 방향을 수정했지만 11.27° 초기 기울기는 남아 있던
5.2초 `grounded/reference.npz`로 학습한 2000iter 정책입니다. 새 학습은 수평 교정한 `stable/`를 사용합니다.
2번의 `policy_reference.npz`는 현재 등록된 1000iter 정책의 접촉 준비 입력입니다.
좌표계·쿼터니언·시간 출처는 각 `manifest.json`과 `grounded/frame.json`에 있습니다.

최근 완료한 2000iter 학습은 **현재 1번 데모**의 수정된 5.2초 `grounded/reference.npz`를 사용했습니다.
해당 정책의 입력은 그대로 보존했으며, 새 수평 입력으로 학습을 다시 수행한 것은 아닙니다.

현재 설정 파일은 새 상대 경로를 사용합니다. 저장된 checkpoint와 NPZ의 과거 경로는 작성 당시 기록으로
보존하고 로더에서 `data/original` → `data/demo1`, `data/current` → `data/demo2`로 연결합니다.
Git 제외 `local/`의 기존 입력 링크도 같은 동작의 새 폴더로 연결했습니다.
Checkpoint 계약 해시에는 두 폴더명의 옛 표기만 정규화하여 사용하므로, 이름 변경만으로 기존 정책이
거부되지 않습니다. 입력 내용의 SHA-256, 모델, 제어·보상 설정 등 나머지 계약 검사는 그대로 유지합니다.

과거 출처/hash 기록을 읽는 검증 코드는 `data.resolve_demo_path(path, root)`로 이전 `data/` 경로를 해석합니다.

## 사용

저장소 루트에서 실행합니다. `PYTHON`과 `PYTHONPATH`는 상위 README의 환경 설정을 사용합니다.

```bash
# 기존 데모의 사람 손/물체 재추출 — 임시 결과로 검증
"$PYTHON" scripts/data.py --output local/results/current_demo/poses.npz

# 가져온 원본을 다시 adapter로 읽기 — 기존 파일을 덮어쓰지 않음
"$PYTHON" scripts/demo.py --source data/demo1/raw/human_demo.npz \
  --output local/results/original_demo_import/input

# 현재 1번 기하 궤적의 팔 IK
"$PYTHON" scripts/ik.py solve --config config/ik_demo1.json \
  --output local/results/ik/demo1_stable

# 1번 데모로 새 학습을 시작할 때의 설정 (실행 시 학습 시작)
"$PYTHON" scripts/policy.py --mode train --headless --skip-evaluation \
  --config config/policy_original.json --output local/results/policy/original_new
```

원본 camera annotation과 리타게팅은 선언된 30Hz, 1.3초입니다. 균등 감속한 `grounded/`의 5.2초 자료를
수평 교정한 현재 입력이 `stable/`입니다. camera 입력에서 새로 리타게팅·grounding하면 자동으로
5.2초가 되지는 않으므로 시간 변환도 명시적으로 적용해야 합니다.

수평 교정은 기존 감속 입력에서 별도로 생성합니다. 기존 파일을 덮어쓰지 않으므로 재생성할 때에는
새 출력 폴더를 지정합니다. 손·물체의 상대 자세, 관절, frame ID와 시간은 유지합니다.

```bash
"$PYTHON" scripts/ground.py --stabilize \
  --poses data/demo1/grounded/poses.npz --reference data/demo1/grounded/reference.npz \
  --output local/results/demo1_stable_regenerated
```

`frame.json`은 고정 좌표 변환과 추가 프레임별 Z 보정을 구분해 기록합니다. RGB 비교용 역변환도
이 보정을 먼저 제거합니다. 시뮬레이션 중 캔을 고정하거나 위치를 계속 덮어쓰는 기능은 아닙니다.
