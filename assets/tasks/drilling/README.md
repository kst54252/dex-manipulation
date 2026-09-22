# Drilling assets

- `drill.usda`: 드릴·비트·트리거의 실제 형상, collision, 질량·관성, 관절.
- `fixture.usda`: 나사와 고정 지그의 형상·collision·고정 관계.

USD의 상대 참조 파일도 이 폴더 안에 함께 둡니다.
비트 끝·회전축·손잡이·트리거 접촉면의 local frame을 `config/tasks/drilling/object.json`에 연결합니다.
트리거 작동 변위/힘, 회전 속도·토크는 실제 도구 정보로 설정합니다.
공유 로봇 USD는 기존 `assets/USD/`를 사용합니다.
