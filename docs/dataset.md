# 데이터 출처

기존 `can_pick` 데모의 사람 손 21개 3D 키포인트와 물체 6D pose는 **DexYCB** 데이터셋을 사용합니다.

- [공식 데이터·다운로드](https://dex-ycb.github.io/)
- [DexYCB Toolkit과 annotation 설명](https://github.com/NVlabs/dex-ycb-toolkit)
- 데이터 라이선스: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- 논문: Yu-Wei Chao et al., *DexYCB: A Benchmark for Capturing Hand Grasping of Objects*, CVPR 2021.

| 데모 | 입력 | Frame ID |
|---|---|---|
| demo1 | `20200709_143747_left`, 카메라 `839512060362`의 전처리 annotation | 12~51 |
| demo2 | 카메라 `839512060362`의 RGB·label | 15~41 |

원본은 `data/can_grasping/demo1/raw/`, `data/can_grasping/demo2/raw/`에 두며 이미지·NPZ는 Git에서 제외합니다.
`manifest.json`의 SHA-256으로 입력 파일을 대응시킵니다.
리타게팅·좌표변환·프레임 선택·시간 조정·접촉 reference는 프로젝트의 파생 처리입니다.
원본과 파생 데이터에는 데이터셋의 출처·이용 조건을 적용합니다.

직접 촬영한 작업 데모는 `data/<task>/demoN/`에 둡니다. 원본 영상·프레임·annotation은 `raw/`에 보관하며 Git에서 제외합니다.
촬영 출처, 실제 프레임 시간, 보정값, mesh·모델 hash와 변환 과정은 `manifest.json`에 기록합니다.
EgoPHI 접촉·힘은 모델 추정 라벨이며 측정값이나 DexYCB 정답으로 표시하지 않습니다.
[RGB 데이터 제작](dataset_capture.md) · [EgoPHI·HaMeR 출처](PROVENANCE.md#rgb-데이터-제작)

## 인용

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
