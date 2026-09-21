# Dataset attribution: DexYCB

이 프로젝트의 사람 손 21개 3D 키포인트와 물체 6D pose 데모는 **DexYCB** 데이터셋에 기반합니다.
RGB·원본 annotation은 이 프로젝트에서 촬영하거나 제작한 데이터가 아닙니다.

- 공식 데이터셋 및 다운로드: [DexYCB](https://dex-ycb.github.io/)
- 논문: Yu-Wei Chao et al., *DexYCB: A Benchmark for Capturing Hand Grasping of Objects*, CVPR 2021.
- annotation 설명: [공식 DexYCB Toolkit](https://github.com/NVlabs/dex-ycb-toolkit)
- 데이터 라이선스: 공식 배포 페이지에 명시된 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  데이터와 그 파생물의 출처·조건은 프로젝트 코드와 구분합니다.

## 저장소에서 사용하는 부분

| 데모 | 입력 출처 | 선택 프레임 | 현재 재생 길이 |
|---|---|---|---|
| `demo1` | DexYCB `20200709_143747_left`, 카메라 `839512060362`; 기존 프로젝트에서 전처리한 사람 annotation을 가져옴 | 12~51, 40개 | 5.2초 |
| `demo2` | DexYCB RGB·label, 카메라 `839512060362`; 로컬에 남긴 시퀀스의 전체 식별자는 확인되지 않음 | 15~41, 27개 | 5.2초 |

데모 번호는 이 프로젝트 내부 이름입니다. 재생 5.2초는 시뮬레이션용 시간 조정이며 원본 촬영 속도라고
주장하지 않습니다. demo1의 오른손 변환은 전처리 metadata에 명시된 변환을 사용합니다.
로봇 손목·관절 리타게팅, 바닥 좌표 보정, 프레임 선택 및 접촉 reference 준비는 이 프로젝트의
파생 처리입니다. 현재 복합 원통 캔 치수도 사용자가 지정한 시뮬레이션 형상으로, 원본 물체의
실측 형상이라고 주장하지 않습니다. 파일별 SHA-256·역할과 변환 기록은 각 데모의 `manifest.json`,
`grounded/frame.json` 및 demo1의 `stable/frame.json`에 있습니다.

## Raw 데이터는 로컬에 보관

`data/demo1/raw/`, `data/demo2/raw/`의 이미지와 NPZ는 Git에서 제외합니다. 특히 demo2의
`color_*.jpg`와 `labels_*.npz`는 업로드하지 않습니다. 로컬 원본은 삭제하지 않습니다.
Git에는 배치 안내 README, 출처·hash metadata, 실행에 사용하는 소규모 파생 pose/reference만 둡니다.

원본을 다시 추출하거나 RGB를 비교하려면 공식 배포처에서 받은 대응 데이터를 각 `raw/`에 배치합니다.
demo2는 `color_000015.jpg`~`color_000041.jpg`, `labels_000015.npz`~`labels_000041.npz`를 사용합니다.
전체 시퀀스 식별자가 없으므로 임의의 DexYCB 시퀀스를 같은 데모라고 대체하지 말고,
manifest의 SHA-256과 대조합니다. 원본이 없는 새 clone에서도 포함된 파생 궤적으로 FK·IK·리타게팅
결과 재생 입력을 읽을 수 있습니다. 학습된 정책 가중치는 별도 로컬 파일이 필요합니다.

이전 `data/raw/`의 추적 파일은 현재 main에서 제거합니다. 일반적인 삭제 커밋은 과거 Git 이력의
데이터까지 지우지는 않으며, 이 작업에서는 공유 이력을 재작성하지 않습니다.

## Citation

```bibtex
@inproceedings{chao2021dexycb,
  author = {Yu-Wei Chao and Wei Yang and Yu Xiang and Pavlo Molchanov and
            Ankur Handa and Jonathan Tremblay and Yashraj S. Narang and
            Karl Van Wyk and Umar Iqbal and Stan Birchfield and Jan Kautz and Dieter Fox},
  title = {{DexYCB}: A Benchmark for Capturing Hand Grasping of Objects},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}
```
