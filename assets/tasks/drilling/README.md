# Drilling assets

- `drill.obj`: RGB 데이터 제작용 삼각형 mesh. OBJ 대신 단일 PLY/STL mesh도 설정할 수 있습니다.
- `drill.usda`: 로봇 시뮬레이션용 형상·collision·질량·관성.

실제 드릴을 측정해 `config/tasks/drilling/capture.json`의 `mesh_unit_to_m`, `mesh_to_object`, `measurement_source`를 지정합니다.
`mesh_to_object`는 mesh를 metre로 변환한 뒤 적용하는 4×4 SE(3) 변환입니다. 크기를 pose로 추정하지 않습니다.
드릴 집기 촬영에서는 부품이 움직이지 않는 하나의 rigid object로 취급합니다. 나사·지그·트리거 물성은 필요하지 않습니다.

USD를 추가할 때 상대 참조 파일도 이 폴더 안에 함께 둡니다.
공유 로봇 USD는 기존 `assets/USD/`를 사용합니다.
