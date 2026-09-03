# VisionAce Improved

**원본 이미지 라벨링과 자동 라벨링을 개선한 별도 Windows 배포판**입니다. 기준은 [VisionAce v1.8.1](https://github.com/NS-DDC/claude_code/releases/tag/v1.8.1)이며, 이 저장소의 `improvements/visionace-v1.9` 브랜치와 `v1.9.0-improved.3` 릴리스로 구분합니다. [원본 출처](docs/UPSTREAM.md) · [변경 이력](CHANGELOG.md)

## Windows에서 바로 실행

1. [개선판 릴리스](https://github.com/NS-DDC/DL-bbox-mask-TrainTool/releases/tag/v1.9.0-improved.3)에서 `VisionAce-Improved-v1.9.0-improved.3-Windows-x64-CPU.zip`을 받습니다.
2. ZIP **전체를 압축 해제**하고 `VisionAce-Improved.exe`를 실행합니다.
3. `_internal` 폴더는 exe 옆에 그대로 둡니다. Python 설치는 필요 없습니다.

Windows 10/11 64비트, CPU용입니다. 추론 라이브러리를 포함하므로 exe 하나만 따로 복사하면 실행되지 않습니다. 모델 가중치는 별도로 준비합니다. 설정과 로그는 `%USERPROFILE%\.visionace-improved\`에 저장됩니다.

## 원본 이미지로 라벨링

`Ctrl+O`로 이미지 폴더를 열고 클래스를 추가한 뒤 `W`로 박스, `E`로 마스크를 그립니다. 화면 배율과 상관없이 좌표는 **저장된 원본 이미지 크기**를 기준으로 합니다.

- `Ctrl+1`: 원본 픽셀 배율. `Ctrl+0`: 화면에 맞춤. 휠: 확대·축소. 중간 버튼: 이동. 분할 모드에서는 휠로 브러시 크기, `Ctrl+휠`로 배율을 조절합니다.
- `Ctrl+S`: 열어서 편집한 이미지들의 라벨 저장. `S`: 현재 이미지 저장 후 다음으로.
- `A`, `D`, `X`: 현재 미저장 편집을 버리고 이전/다음 이미지로 이동.
- `Ctrl+Z`, `Ctrl+Y`: 실행 취소·다시 실행. 자동 라벨 적용도 이미지별로 되돌릴 수 있습니다.
- `F`: 현재 이미지를 저장한 뒤 이번 목록에서만 숨깁니다. 폴더를 다시 열면 나타납니다. 학습 데이터셋 제외 설정은 바꾸지 않습니다.

기본 저장은 **라벨만 저장**하며 원본을 덮어쓰거나 다시 인코딩하지 않습니다. `설정 → 저장할 때 원본을 images/에 그대로 복사`를 켜면 원본 바이트를 그대로 복사합니다. 다른 파일이 이미 있으면 덮어쓰지 않고 오류를 표시합니다.

JPEG/PNG/BMP/TIF/TIFF와 한글 경로를 지원합니다. EXIF 방향을 자동 회전하지 않아 화면과 추론이 같은 원본 픽셀 축을 사용합니다. 16비트 TIFF 파일은 원본을 그대로 보존하지만 화면과 모델 입력은 8비트 BGR 버퍼입니다. 다중 페이지 TIFF 전체 페이지 편집이나 타일 기반 초대형 이미지 처리는 포함하지 않습니다.

```text
원본폴더/
├── wafer.tif                       # 수정하지 않는 원본
├── labels/
│   ├── wafer.txt                   # YOLO 원본 크기 기준 정규화 좌표
│   └── .visionace-project.json     # 클래스 번호·이름·색상
├── gt_image/클래스명/wafer.png       # 원본 WxH 픽셀 마스크
└── images/wafer.tif                # 원본 복사를 켰을 때만 생성
```

클래스 번호가 바뀌지 않도록 프로젝트별 목록을 저장합니다. `classes.txt`가 있으면 줄 순서를 클래스 ID로 가져옵니다. 이름이 같은 `wafer.jpg`와 `wafer.png`는 하나의 YOLO 파일을 공유하게 되므로 함께 열지 못하게 검사합니다. 클래스 삭제는 번호 변경을 방지하기 위해 라벨이 없는 프로젝트의 마지막 클래스만 허용합니다.

## 자동 라벨링

1. `파일 → 모델 불러오기`에서 **로컬 모델 파일**을 선택합니다.
2. `AUTO`를 선택하거나 `YOLO`/`RT-DETR`를 직접 지정합니다. `best.pt`처럼 파일명에 모델 종류가 없어도 구조로 판별합니다.
3. `도구 → 자동 라벨링`에서 신뢰도, 모델 입력 크기, 장치, 라벨 종류와 범위를 선택합니다.
4. 기본값은 기존 라벨이 있는 이미지 건너뛰기입니다. 필요하면 추가 또는 교체를 고릅니다.
5. 결과를 검토하고 `Ctrl+S`로 저장합니다. 오류는 대화상자의 상세 목록과 로그에서 확인합니다.

추론 크기는 모델 입력 크기입니다. 화면 해상도나 원본 파일을 바꾸지 않습니다. 아주 작은 객체는 모델 입력에서 작아지므로 크기를 늘릴 수 있지만 메모리 사용도 증가합니다. 정사각형 마스크 패딩을 제거하고 원본 좌표로 복원합니다. 분할 모델은 폴리곤 또는 픽셀 마스크로 결과를 받으며, 내부 구멍을 유지하려면 픽셀 마스크를 선택하세요.

| 모델 | 이 배포판의 지원 범위 |
|---|---|
| 학습된 Ultralytics YOLO `.pt` | 탐지·인스턴스 분할 |
| 학습된 Ultralytics RT-DETR `.pt` | 탐지, 사용자 파일명 지원 |
| Keras `.h5`/`.keras` | 별도 소스 환경에서 NHWC 분할 확률 모델만 지원; exe에는 TensorFlow 미포함 |
| DINOv3 `.pth` 백본 | 직접 추론 미지원. 별도 탐지/분할 헤드·설정·어댑터 필요 |
| Grounding DINO / Paddle / Hugging Face RT-DETR / ONNX / 임의 state_dict | 해당 형식의 어댑터 미포함 |

DINOv3와 Grounding DINO는 서로 다른 모델입니다. [DINOv3 공식 detector 구현](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/detectors.py)은 백본과 검출 헤드를 따로 구성합니다. 백본 파일을 YOLO처럼 불러오는 경로는 성공으로 표시하지 않습니다.

exe의 `auto` 장치는 CPU입니다. 소스 환경에 CUDA PyTorch를 설치한 경우 `auto`는 사용 가능한 CUDA를 선택하고, `0` 또는 `cuda:0`으로 직접 지정할 수 있습니다. 사용 불가능한 GPU를 선택하면 오류를 보여 줍니다. 모델 로드 실패 시 이전 모델은 유지되며, 추론 실패로 기존 라벨을 지우지 않습니다.

## 소스 실행과 빌드

이 작업에서는 사용자 PC에서 아래 명령을 실행하지 않았습니다. 필요할 때 별도의 개발/검증 PC에서 사용하세요.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
python main.py
```

선택적인 Keras 분할 지원은 해당 소스 가상환경에 TensorFlow/Keras를 별도로 설치해야 합니다. 입력은 NHWC, RGB 또는 grayscale, float32/255이며 출력은 픽셀별 확률입니다. 임의의 분류 모델이나 다른 전처리의 체크포인트를 자동 변환하지 않습니다.

Windows exe 빌드는 GitHub Actions의 `Windows portable CPU release`에서 수행합니다. `improvements/**` 브랜치는 회귀 검사, `v1.9.0*` 태그는 검사 후 exe 빌드·릴리스 업로드를 수행합니다. 수동 `workflow_dispatch`는 릴리스용 ZIP을 Actions artifact로 만듭니다. 자세한 파일은 [workflow](.github/workflows/windows-release.yml), [spec](visionace.spec), [requirements](requirements.txt)를 참고하세요.

## 검증 경계

사용자의 PC에서는 테스트·앱 실행·모델 추론·패키지 설치·exe 빌드를 하지 않았습니다. 배포는 GitHub Windows 러너의 회귀 검사와 exe 기동·Qt 화면 렌더링·네이티브 라이브러리 검사가 성공한 경우에만 진행됩니다. 결과는 ZIP의 `verification/`, 빌드 출처는 `build-provenance.json`, 정확한 패키지 버전은 `requirements-resolved.txt`에서 확인하세요.

사용자의 실제 학습 모델과 실제 이미지에 대한 정확도, GPU 동작, 장시간 대용량 작업은 검증하지 않았습니다. 저장은 파일별 원자적 교체이며 여러 파일 전체를 하나로 되돌리는 트랜잭션은 아닙니다. 문제가 있으면 로그와 모델 형식, 이미지 크기, 재현 순서를 남겨 주세요.

의존성 고지는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있습니다. 원본 README의 MIT 표시는 유지하되, 번들에 포함한 각 라이브러리의 라이선스는 별도로 적용됩니다.
