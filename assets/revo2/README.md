# Revo2 자산과 키포인트

`regrind-upload`에서 가져온 USD 자산과 21개 semantic keypoint 정의입니다.
데이터 처리, retargeting, FK/IK, replay, RL 코드는 포함하지 않습니다.

## USD 진입 파일

프로젝트 루트를 기준으로 합니다. 하위 USD와 텍스처의 상대 경로를 유지하세요.

| 경로 | 용도 |
| --- | --- |
| `assets/USD/revo2_right/revo2_right.usda` | Revo2 오른손 및 키포인트 Xform |
| `assets/USD/revo2_floating.usda` | Floating Revo2 |
| `assets/USD/rb3_730es_u/rb3_730es_u.usda` | RB3-730 팔 |
| `assets/USD/rb3_revo2.usd` | RB3 + Revo2 조립 모델 |
| `assets/USD/rb3_revo2_vertical.usda` | 수직 어댑터 장착 모델 |
| `assets/USD/revo2_vertical_adapter/revo2_vertical_adapter.usd` | 수직 어댑터 |
| `assets/USD/tuna_fish_can_rigid.usda` | 참치캔 rigid-body wrapper |
| `assets/007_tuna_fish_can/textured_simple.usd` | 참치캔 메시 및 재질 |

`assets/USD/*/payloads/`에는 모델의 형상·관절·물리·재질 레이어가 들어 있습니다.
참치캔은 `assets/007_tuna_fish_can/textures/texture_map.png`를 참조합니다.
어댑터의 `OmniPBR.mdl`은 Isaac Sim/Omniverse 런타임 재질 의존성입니다.

## 키포인트 좌표와 순서

정의 파일은 이 디렉터리의 `revo2_keypoints.json`입니다.
원본 `tools/revo2_kinematics/revo2_keypoints.json`을 값 변경 없이 복사했습니다.
원본의 패키지 내부 JSON 사본과도 동일합니다.

- JSON 키 `"0"`–`"20"`을 숫자로 정렬해서 사용합니다.
- `name`: semantic keypoint 이름.
- `parent_link`: 키포인트 좌표가 정의된 부모 링크.
- `xyz`: 해당 부모 링크 좌표계의 위치, 단위는 m입니다. 월드 좌표가 아닙니다.
- `prim_path`: 단독 Revo2 USD의 키포인트 경로입니다. 다른 prim 아래에 모델을
  배치하면 경로 접두부가 달라질 수 있으므로 링크/키포인트 이름으로 대응시킵니다.
- 월드 좌표는 현재 부모 링크의 월드 변환에 `xyz`를 적용해서 계산합니다.
- 이 순서는 thumb부터 나열하는 sequential MANO21 순서와 다릅니다.
- 21개 키포인트는 기구학적 출력입니다. 독립 구동 관절은 6개입니다.

| 인덱스 | 이름 | 부모 링크 |
| --- | --- | --- |
| 0 | kp_00_wrist | right_hand_base_link |
| 1 | kp_01_index_mcp | right_hand_base_link |
| 2 | kp_02_index_pip | right_index_proximal_link |
| 3 | kp_03_index_dip | right_index_distal_link |
| 4 | kp_04_middle_mcp | right_hand_base_link |
| 5 | kp_05_middle_pip | right_middle_proximal_link |
| 6 | kp_06_middle_dip | right_middle_distal_link |
| 7 | kp_07_little_mcp | right_hand_base_link |
| 8 | kp_08_little_pip | right_pinky_proximal_link |
| 9 | kp_09_little_dip | right_pinky_distal_link |
| 10 | kp_10_ring_mcp | right_hand_base_link |
| 11 | kp_11_ring_pip | right_ring_proximal_link |
| 12 | kp_12_ring_dip | right_ring_distal_link |
| 13 | kp_13_thumb_mcp | right_thumb_proximal_link |
| 14 | kp_14_thumb_pip | right_thumb_proximal_link |
| 15 | kp_15_thumb_dip | right_thumb_distal_link |
| 16 | kp_16_thumb_tip | right_thumb_touch_link |
| 17 | kp_17_index_tip | right_index_touch_link |
| 18 | kp_18_middle_tip | right_middle_touch_link |
| 19 | kp_19_ring_tip | right_ring_touch_link |
| 20 | kp_20_little_tip | right_pinky_touch_link |

기존 6개 구동 관절 순서는 다음과 같습니다.

1. `right_thumb_metacarpal_joint`
2. `right_thumb_proximal_joint`
3. `right_index_proximal_joint`
4. `right_middle_proximal_joint`
5. `right_ring_proximal_joint`
6. `right_pinky_proximal_joint`

기존 기구학 정의에서 엄지 distal은 thumb proximal을 1.0배로,
나머지 네 손가락 distal은 각 proximal을 1.155배로 따릅니다(offset 0).
원본의 관절 한계·mimic·물리 설정은 USD 레이어에 보존되어 있습니다.

## 복사 검증

USD와 텍스처 및 키포인트 JSON의 원본/사본 SHA-256 일치를 확인합니다.
21개 키포인트의 부모 링크 및 로컬 좌표는 단독 Revo2 USD와 비교해
일치함을 확인했습니다. 새 데이터 처리 파이프라인이나 Isaac 물리 실행은
이 자산 이관에 포함하지 않습니다.
