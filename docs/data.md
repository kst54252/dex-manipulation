# 데모 데이터

캔 집기는 데모2를 사용합니다. 입력은 사람 손 21개 3D 점과 물체 6D pose입니다. [DexYCB 출처](dataset.md)

| 항목 | 데모2 |
|---|---|
| 폴더 | `data/can_grasping/demo2/` |
| 원본 frame ID | 15~41, 27자세 |
| 기하 궤적 기준 길이 | 5.2초 |
| 기하 재생 입력 | `grounded/reference.npz` |
| 리타게팅 설정 | `config/tasks/can_pick/retargeting_demo2.json` |
| 팔 설정 | `config/tasks/can_pick/ik_demo2.json` |
| 기본 플로팅 학습 입력 | `local/references/contact/demo2.npz` |

정책은 checkpoint에 저장된 시간축을 사용하며 `run.sh --speed`는 그 시간축에 적용하는 배속입니다.

| 파일 | 내용 |
|---|---|
| `raw/` | 원본 이미지·annotation, Git 제외 |
| `poses.npz` | 카메라 좌표의 사람 손·물체 |
| `retargeted.npz` | 카메라 좌표의 로봇 손목·관절 |
| `grounded/` | 바닥 좌표의 pose·reference·변환 기록 |
| `policy_reference.npz` | 해당 입력으로 학습한 checkpoint의 기준 궤적 |
| `manifest.json` | 파일 역할·단위·프레임·SHA-256 |

접촉 학습 입력은 [별도 생성](contact_training.md)하며 정책은 저장된 설정의 reference를 읽습니다.
`resolve_demo_path()`는 과거 `data/demo2`, `data/current`를 `data/can_grasping/demo2`로 연결합니다.
기존 checkpoint와 저장된 메타데이터는 유지합니다.

```bash
./run.sh floating retarget 2
./run.sh floating policy 2
./run.sh train floating 2 -i 1000
```
