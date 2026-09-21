# Revo2 자산

`regrind-revo2`에서 가져온 로봇 USD와 semantic 키포인트 정의입니다.
USD 하위 레이어·텍스처의 상대 경로를 유지합니다.

| 진입 파일 | 역할 |
|---|---|
| `assets/USD/revo2_right/revo2_right.usda` | Revo2 오른손 |
| `assets/USD/revo2_floating.usda` | 플로팅 손 |
| `assets/USD/rb3_730es_u/rb3_730es_u.usda` | RB3-730 팔 |
| `assets/USD/rb3_revo2_vertical.usda` | 일자 어댑터로 조립한 RB3 + Revo2 |
| `assets/USD/revo2_vertical_adapter/revo2_vertical_adapter.usd` | 장착 어댑터 |

`revo2_keypoints.json`은 21점의 이름·부모 링크·local XYZ(m)를 정의합니다.
월드 좌표는 부모 링크 변환에 local 좌표를 적용해 계산합니다.
사람 손과는 배열 번호가 아닌 semantic 이름으로 대응시킵니다.
독립 관절 6개와 가동 관절 11개의 관계는 USD mimic으로 정의합니다.

[FK·리타게팅](../../docs/retargeting.md) · [IK](../../docs/ik.md)
