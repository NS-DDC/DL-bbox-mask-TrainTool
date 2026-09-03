# 원본과 개선판 구분

- 기준 릴리스: [NS-DDC/claude_code v1.8.1](https://github.com/NS-DDC/claude_code/releases/tag/v1.8.1)
- 기준 커밋: `dc8af427a1efba14a822a683484122f0983806b3`
- 가져온 범위: 해당 커밋의 `DL-Tool/` 하위 소스만.
- 개선판 저장소: [NS-DDC/DL-bbox-mask-TrainTool](https://github.com/NS-DDC/DL-bbox-mask-TrainTool)
- 개선판 브랜치: `improvements/visionace-v1.9`
- 개선판 태그: `v1.9.0-improved.2`
- 대상 저장소의 기존 main 기준: `2d24e64641130bdee254fb281a5ad0ee396c1404`

원본 `claude_code`의 브랜치·태그·릴리스는 수정하지 않았습니다. 다른 앱의 소스와 원본 저장소의 Git 이력을 개선판에 옮기지 않았습니다. 기존 main과 개선판은 브랜치로 구분합니다.

개선판 사용자 설정은 `.visionace-improved/`에 저장하므로 이전 `.visionace/` 설정과 분리됩니다. 프로젝트의 `labels/.visionace-project.json`은 클래스 목록과 순서를 저장합니다.

## 구현 확인에 사용한 공식 자료

- [Ultralytics 모델 클래스와 RT-DETR 구조 판별](https://docs.ultralytics.com/reference/models/yolo/model/)
- [RT-DETR 전처리 및 원본 크기 좌표 복원](https://docs.ultralytics.com/reference/models/rtdetr/predict/)
- [Results와 원본 좌표 마스크](https://docs.ultralytics.com/reference/engine/results/)
- [DINOv3의 별도 detector와 backbone 구성](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/detectors.py)
- [PyTorch와 torchvision 호환 설치](https://docs.pytorch.org/get-started/previous-versions/)

실제 사용자 모델 파일이 제공되지 않았으므로 체크포인트별 전처리·클래스 의미·정확도는 검증 범위에 포함하지 않았습니다.
