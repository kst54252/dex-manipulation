# 데이터 출처: DexYCB

사람 손 21개 3D 키포인트와 물체 6D pose는 **DexYCB** 데이터셋을 사용합니다.

- [공식 데이터·다운로드](https://dex-ycb.github.io/)
- [DexYCB Toolkit과 annotation 설명](https://github.com/NVlabs/dex-ycb-toolkit)
- 데이터 라이선스: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- 논문: Yu-Wei Chao et al., *DexYCB: A Benchmark for Capturing Hand Grasping of Objects*, CVPR 2021.

| 데모 | 입력 | Frame ID |
|---|---|---|
| demo1 | `20200709_143747_left`, 카메라 `839512060362`의 전처리 annotation | 12~51 |
| demo2 | 카메라 `839512060362`의 RGB·label | 15~41 |

원본은 `data/demo1/raw/`, `data/demo2/raw/`에 두며 이미지·NPZ는 Git에서 제외합니다.
`manifest.json`의 SHA-256으로 입력 파일을 대응시킵니다.
리타게팅·좌표변환·프레임 선택·시간 조정·접촉 reference는 프로젝트의 파생 처리입니다.
원본과 파생 데이터에는 데이터셋의 출처·이용 조건을 적용합니다.

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
