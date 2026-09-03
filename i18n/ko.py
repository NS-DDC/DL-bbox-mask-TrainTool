"""Korean translations for VisionAce."""

KO = {
    # Application
    "app_title": "VisionAce - 딥러닝 라벨링 & 학습 도구  |  by KNS",

    # Menu - File
    "menu_file": "파일(&F)",
    "action_open_folder": "이미지 폴더 열기",
    "action_load_model": "모델 로드 (.pt, .h5)",
    "action_save_labels": "라벨 저장",
    "action_export_masks": "바이너리 마스크 내보내기",
    "action_exit": "종료",

    # Menu - Edit
    "menu_edit": "편집(&E)",
    "action_undo": "실행 취소",
    "action_redo": "다시 실행",
    "action_delete_label": "선택 라벨 삭제",

    # Menu - Tools
    "menu_tools": "도구(&T)",
    "action_auto_label": "오토 라벨링...",
    "action_training": "학습...",

    # Menu - Settings
    "menu_settings": "설정(&S)",
    "action_set_label_dir": "라벨 폴더 지정...",
    "action_lang_ko": "한국어",
    "action_lang_en": "English",

    # Label folder settings
    "label_dir_title": "라벨 폴더",
    "label_dir_message": "커스텀 라벨 폴더를 지정하시겠습니까?\n\n예: 커스텀 폴더 선택\n아니오: 기본 labels/ 하위폴더 사용",
    "label_dir_no_project": "먼저 이미지 폴더를 열어주세요.",
    "label_dir_error": "라벨 폴더 설정에 실패했습니다.",
    "select_label_folder": "라벨 폴더 선택",

    # Toolbar
    "tool_detection": "Detection (BBox) [W]",
    "tool_segmentation": "Segmentation (Mask) [E]",
    "tool_select": "선택 / 이동 [Q]",

    # File list panel
    "file_panel_title": "이미지 파일",
    "file_no_folder": "폴더가 로드되지 않았습니다",
    "file_count": "{count}개 이미지",

    # Label list panel
    "label_panel_title": "라벨",
    "label_classes_title": "클래스",
    "label_instances_title": "인스턴스",
    "label_add_class": "클래스 추가",
    "label_remove_class": "클래스 삭제",
    "label_class_name": "클래스 이름:",
    "label_no_labels": "이 이미지에 라벨이 없습니다",
    "label_show_all": "모두 보기",
    "label_hide_all": "모두 숨기기",

    # BBox coordinate format
    "bbox_coord_format": "좌표 표시",
    "bbox_coord_absolute": "절대좌표 (px)",
    "bbox_coord_relative": "상대좌표 (0~1)",
    "bbox_info_absolute": "[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]",
    "bbox_info_relative": "[{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}]",
    "polygon_info_absolute": "{n}점",
    "polygon_info_relative": "{n}점 (norm)",

    # Canvas
    "canvas_no_image": "라벨링을 시작하려면 이미지 폴더를 여세요",
    "canvas_zoom": "확대: {zoom}%",

    # Auto labeling dialog
    "auto_label_title": "오토 라벨링",
    "auto_label_model": "모델:",
    "auto_label_confidence": "신뢰도 임계값:",
    "auto_label_scope": "범위:",
    "auto_label_current": "현재 이미지",
    "auto_label_all": "전체 이미지",
    "auto_label_start": "시작",
    "auto_label_cancel": "취소",
    "auto_label_progress": "처리 중 {current}/{total}...",
    "auto_label_progress_pct": "{current} / {total}  ({pct}%)",
    "auto_label_time_info": "속도: {speed} img/s  |  남은 시간: {eta}  |  경과: {elapsed}",
    "auto_label_elapsed": "총 소요 시간: {elapsed}",
    "auto_label_complete": "오토 라벨링 완료: {count}개 라벨 생성",
    "auto_label_no_model": "모델이 로드되지 않았습니다. 먼저 모델을 로드해주세요.",
    "auto_label_score_threshold": "점수 임계값:",
    "auto_label_score_threshold_tooltip": (
        "추론 후 적용되는 후처리 신뢰도 필터입니다. "
        "이 값보다 낮은 점수의 탐지 결과는 제거됩니다. "
        "더 엄격한 필터링을 원하면 신뢰도보다 높게 설정하세요."
    ),
    "auto_label_infer_size": "추론 이미지 크기 (px):",
    "auto_label_infer_size_tooltip": (
        "YOLO / RT-DETR 추론 시 사용할 정사각형 이미지 크기 (예: 480). "
        "좌표는 자동으로 원본 이미지 해상도로 역변환됩니다."
    ),
    "auto_label_confidence_tooltip": (
        "모델에 전달되는 최소 신뢰도 (NMS 임계값)입니다. "
        "낮게 설정하면 더 많은 탐지 결과를 얻을 수 있으며, "
        "점수 임계값으로 추가 필터링하세요."
    ),
    "auto_label_device": "디바이스:",
    "auto_label_device_tooltip": (
        "추론에 사용할 디바이스입니다 (YOLO/RT-DETR 전용). "
        "auto: GPU 사용 가능 시 자동 선택, "
        "cpu: CPU 사용, "
        "0: GPU 0번. 직접 입력도 가능합니다 (예: 1, mps)."
    ),

    # Training dialog
    "training_title": "모델 학습",
    "training_model_type": "모델 유형:",
    "training_base_model": "베이스 모델:",
    "training_dataset": "데이터셋 경로:",
    "training_browse": "찾아보기...",
    "training_epochs": "에포크:",
    "training_batch_size": "배치 크기:",
    "training_img_size": "이미지 크기:",
    "training_lr": "학습률:",
    "training_device": "디바이스:",
    "training_start": "학습 시작",
    "training_stop": "학습 중지",
    "training_close": "닫기",
    "training_log": "학습 로그",
    "training_progress": "에포크 {epoch}/{total} - 손실: {loss:.4f}",
    "training_complete": "학습 완료! 최적 모델 저장 위치: {path}",
    "training_generate_yaml": "data.yaml 자동 생성",
    "training_classes": "클래스:",
    "training_train_path": "학습 경로:",
    "training_val_path": "검증 경로:",

    # Status bar
    "status_ready": "준비",
    "status_mode": "모드: {mode}",
    "status_image_info": "{filename} | {width}x{height}",
    "status_cursor": "({x}, {y})",
    "status_model_loaded": "모델 로드됨: {name}",
    "status_no_model": "모델 없음",

    # General
    "ok": "확인",
    "cancel": "취소",
    "yes": "예",
    "no": "아니오",
    "error": "오류",
    "warning": "경고",
    "info": "알림",
    "confirm_delete": "정말 삭제하시겠습니까?",
    "select_folder": "이미지 폴더 선택",
    "select_model": "모델 파일 선택",
    "model_files": "모델 파일 (*.pt)",

    # Help dialog
    "help_title": "도움말 - VisionAce",
    "help_workflow_title": "작업 흐름",
    "help_shortcuts_title": "키보드 단축키",
    "help_tools_title": "도구 사용법",
    "help_tips_title": "팁",
    "help_workflow_step1": "1. 이미지 폴더 열기 (Ctrl+O)",
    "help_workflow_step2": "2. 클래스 추가 및 색상 설정",
    "help_workflow_step3": "3. 모드 선택: Detection(B) 또는 Segmentation(S)",
    "help_workflow_step4": "4. 라벨링 작업 수행",
    "help_workflow_step4_bbox": "   • BBox: 드래그로 박스 그리기",
    "help_workflow_step4_seg": "   • Segmentation: 브러시로 마스크 그리기",
    "help_workflow_step5": "5. 작업 안할 이미지는 X키로 건너뛰기",
    "help_workflow_step6": "6. 저장 (Ctrl+S) 또는 자동저장",
    "help_workflow_result": "결과 폴더:",
    "help_workflow_result_images": "   • images/: 작업한 원본 이미지",
    "help_workflow_result_labels": "   • labels/: BBox/Polygon YOLO 라벨",
    "help_workflow_result_gtimage": "   • gt_image/: Segmentation 마스크",
    "help_shortcut_open": "Ctrl+O : 이미지 폴더 열기",
    "help_shortcut_save": "Ctrl+S : 라벨 저장 (GT 마스크는 PNG로 고정 저장)",
    "help_shortcut_undo": "Ctrl+Z : 실행 취소",
    "help_shortcut_redo": "Ctrl+Y : 다시 실행",
    "help_shortcut_delete": "Delete : 선택 라벨 삭제",
    "help_shortcut_prev": "A : 이전 이미지 (저장 안함)",
    "help_shortcut_next_save": "S : 저장 후 다음 이미지",
    "help_shortcut_next_no_save": "D : 저장 안하고 다음 이미지",
    "help_shortcut_exclude": "F : 학습에서 제외 (이미지/라벨/GT 삭제)",
    "help_shortcut_quit": "Ctrl+Q : 종료",
    "help_shortcut_select": "Q : 선택/편집 모드",
    "help_shortcut_bbox": "W : Detection(BBox) 모드",
    "help_shortcut_segmentation": "E : Segmentation(Mask) 모드",
    "help_shortcut_enter": "Enter : 현재 폴리곤/마스크 완료",
    "help_shortcut_esc": "Esc : 현재 작업 취소",
    "help_shortcut_f1": "F1 : 도움말 패널 토글",
    "help_shortcut_skip": "X : 다음 이미지로 건너뛰기 (자동저장 무시)",
    "help_tool_select": "선택 모드 (V): 라벨을 클릭하여 선택하고, 핸들을 드래그하여 크기 조절 및 이동할 수 있습니다.",
    "help_tool_bbox": "Detection 모드 (B): Rectangle 모드에서는 드래그로 박스를 그립니다. Polygon 모드에서는 클릭으로 점을 찍고 Enter나 우클릭으로 완료합니다.",
    "help_tool_segmentation": "Segmentation 모드 (E): 좌클릭 드래그로 브러시 그리기, 우클릭으로 지우기. Ctrl+클릭으로 폴리곤 그리기 (Enter/우클릭으로 완료). 마스크는 항상 PNG로 저장됩니다.",
    "help_tip_zoom": "Segmentation 모드: 마우스 휠로 브러시 크기 조절, Ctrl+휠로 확대/축소. 다른 모드: 마우스 휠로 확대/축소.",
    "help_tip_pan": "마우스 중간 버튼(휠 클릭)으로 화면 이동.",
    "help_tip_brush": "브러시 모양(원형/사각형)과 크기를 툴바에서 선택 가능.",
    "help_tip_bbox_mode": "BBox 모드를 Rectangle/Polygon 중 선택 가능.",
    "help_tip_class_color": "클래스 목록에서 더블클릭하면 색상 변경.",
    "help_tip_autosave": "다른 이미지로 이동하면 자동 저장됩니다. A(이전), S(저장O 다음), D(저장X 다음) 키로 빠른 이동.",
    "help_tip_right_click": "폴리곤 그리는 중 우클릭으로 빠르게 완료.",
    "help_tip_save_extension": "GT 마스크(Segmentation)는 노이즈 방지를 위해 항상 PNG(무손실) 형식으로 저장됩니다. 원본 이미지는 원본 확장자 그대로 images/ 폴더에 복사됩니다.",
    "help_tip_recent_folders": "파일 > 최근 폴더 메뉴에서 최근 작업한 폴더를 빠르게 열 수 있습니다.",
    "help_tip_exclude": "F키로 현재 이미지를 학습에서 제외할 수 있습니다. 원본, 라벨, GT 이미지가 모두 삭제됩니다.",
    "help_tip_mask_edit": "SELECT 모드에서 마스크 라벨을 더블클릭하면 기존 마스크를 브러시로 수정할 수 있습니다.",
    "help_tip_import_labels": "파일 > 외부 라벨/GT 가져오기로 다른 곳에서 작업한 라벨과 GT를 현재 프로젝트로 가져올 수 있습니다.",
    "help_tip_label_format": "라벨 파일은 '클래스명 클래스ID 좌표...' 형식으로 저장됩니다. 기존 형식도 호환됩니다.",
    "help_tip_resume_work": "이미지 폴더를 열면 기존 labels/, gt_image/ 작업을 자동으로 불러와 이어서 편집할 수 있습니다. Segmentation 모드에서는 이미지 선택 시 마스크가 자동으로 브러시에 로드되어 바로 수정 가능합니다.",
    "help_shortcut_class_number": "1~9, 0 : 클래스 빠른 전환 (1=첫번째, 0=열번째)",
    "help_shortcut_brush_plus": "+ / 휠 업 (Seg모드) : 브러시 크기 증가 (+5)",
    "help_shortcut_brush_minus": "- / 휠 다운 (Seg모드) : 브러시 크기 감소 (-5)",
    "help_shortcut_zoom_in": "Ctrl+휠 업 / Ctrl++ : 이미지 확대",
    "help_shortcut_zoom_out": "Ctrl+휠 다운 / Ctrl+- : 이미지 축소",
    "help_close": "닫기",

    # Menu - Help
    "menu_help": "도움말(&H)",
    "action_help_dialog": "도움말 대화상자",
    "action_toggle_help_panel": "도움말 패널 표시",

    # Help dock
    "help_dock_title": "도움말 (F1 또는 ? 버튼)",

    # Recent menus
    "menu_recent_dirs": "최근 폴더",
    "menu_recent_models": "최근 모델",
    "menu_recent_none": "(없음)",
    "menu_recent_clear": "목록 지우기",

    # Navigation actions
    "action_prev_image": "이전 이미지 (A)",
    "action_next_save": "저장 후 다음 이미지 (S)",
    "action_next_no_save": "저장 안하고 다음 이미지 (D)",
    "action_exclude_training": "학습에서 제외 (F)",

    # Settings
    "action_set_save_extension": "저장 이미지 확장자 설정...",
    "help_tooltip": "도움말 열기/닫기 (F1)",

    # Save extension
    "save_ext_label": "저장 확장자:",
    "save_ext_original": "원본",
    "save_ext_tooltip": "이미지 저장 시 사용할 확장자",

    # Navigation buttons
    "nav_prev": "◀ 이전 [A]",
    "nav_prev_tooltip": "이전 이미지로 이동 (단축키: A)",
    "nav_next_save": "저장O 다음 [S]",
    "nav_next_save_tooltip": "현재 작업을 저장하고 다음 이미지로 이동 (단축키: S)",
    "nav_next_no_save": "저장X 다음 [D]",
    "nav_next_no_save_tooltip": "현재 작업을 저장하지 않고 다음 이미지로 이동 (단축키: D)",

    # Toolbar labels
    "toolbar_bbox_mode": "BBox 모드:",
    "toolbar_brush": "브러시:",
    "toolbar_finish": "완료 [Enter]",

    # Status messages
    "status_next_no_save": "다음 이미지 (저장 안함)",
    "status_next_with_save": "저장 후 다음 이미지로 이동",
    "status_prev_image": "이전 이미지로 이동",
    "status_skipped": "다음 이미지로 건너뛰기 (저장 안함)",

    # Exclude from training
    "exclude_title": "학습에서 제외",
    "exclude_confirm": "'{name}'을(를) 학습에서 제외하시겠습니까?\n\n다음 항목들이 삭제됩니다:\n- 원본 이미지\n- 라벨 파일 (.txt)\n- GT 이미지 파일들\n- images 폴더의 복사본",
    "exclude_done": "'{name}' 학습에서 제외됨 (삭제 완료)",

    # Folder not found
    "folder_not_found_title": "폴더를 찾을 수 없음",
    "folder_not_found_msg": "폴더가 존재하지 않습니다:\n{path}",
    "model_not_found_title": "모델 파일을 찾을 수 없음",
    "model_not_found_msg": "모델 파일이 존재하지 않습니다:\n{path}",

    # Save extension dialog
    "save_ext_dialog_title": "기본 저장 이미지 확장자 설정",
    "save_ext_dialog_msg": "현재 설정: {current}\n\n자동 저장 시 사용할 이미지 형식을 선택하세요:\n(Ctrl+S로 전체 저장 시에는 별도 선택 가능)",
    "save_ext_use_original": "원본 확장자 사용",
    "save_ext_status": "기본 저장 확장자: {ext}",
    "save_ext_display_original": "원본 확장자",
    "save_ext_toolbar_status": "저장 확장자: {ext}",

    # Import external labels
    "action_import_labels": "외부 라벨/GT 가져오기...",
    "import_select_folder": "외부 라벨 폴더 선택",
    "import_no_project": "먼저 이미지 폴더를 열어주세요.",
    "import_no_data": "선택한 폴더에 labels/ 또는 gt_image/ 폴더가 없습니다.",
    "import_complete": "가져오기 완료: {labels}개 라벨, {gt}개 GT 이미지 복사됨",

    # Mask edit
    "mask_edit_status": "마스크 편집 모드 - 브러시로 수정 후 Enter로 완료",

    # Export mask format
    "export_mask_format_title": "마스크 형식 선택",
    "export_mask_format_message": "다중 라벨 시맨틱 마스크로 내보내시겠습니까?\n예: 픽셀값 = 클래스 ID + 1 (시맨틱 세그멘테이션)\n아니오: 바이너리 마스크 (전경=255, 배경=0)",

    # Help – Data formats section
    "help_formats_title": "데이터 형식 & 불러오기",

    "help_fmt_folder_title": "📁 폴더 구조",
    "help_fmt_folder_body": (
        "이미지 폴더를 열면 다음과 같은 구조로 데이터를 자동 인식합니다:\n"
        "\n"
        "  <이미지폴더>/\n"
        "  ├── image1.jpg       ← 원본 이미지\n"
        "  ├── image2.png\n"
        "  ├── labels/          ← BBox/Polygon YOLO 라벨\n"
        "  │   ├── image1.txt\n"
        "  │   └── image2.txt\n"
        "  ├── gt_image/        ← Segmentation GT 마스크\n"
        "  │   ├── 클래스A/\n"
        "  │   │   ├── image1.png\n"
        "  │   │   └── image2.png\n"
        "  │   └── 클래스B/\n"
        "  │       └── image1.png\n"
        "  └── images/          ← 저장 시 복사된 이미지"
    ),

    "help_fmt_yolo_title": "📄 YOLO 라벨 형식 (labels/*.txt)",
    "help_fmt_yolo_body": (
        "BBox:  <클래스명> <클래스ID> <cx> <cy> <w> <h>  (모두 0~1 정규화)\n"
        "예)    cat 0 0.5 0.4 0.2 0.3\n"
        "\n"
        "Polygon:  <클래스명> <클래스ID> <x1> <y1> <x2> <y2> ...\n"
        "예)       dog 1 0.1 0.2 0.3 0.4 0.5 0.2\n"
        "\n"
        "※ 기존 형식(클래스ID만 첫 번째 열)도 자동 호환됩니다."
    ),

    "help_fmt_gtmask_title": "🖼️ GT 마스크 형식 (gt_image/<클래스명>/)",
    "help_fmt_gtmask_body": (
        "• 바이너리 PNG 이미지 (전경=255, 배경=0)\n"
        "• 파일 이름 = 원본 이미지의 stem (확장자 제외)\n"
        "• 클래스별 하위 폴더에 저장  (gt_image/cat/image1.png)\n"
        "• 항상 PNG로 저장 — JPEG 압축 노이즈 방지\n"
        "• 불러올 때 픽셀값 ≥ 128 이면 전경으로 인식"
    ),

    "help_fmt_import_title": "📥 외부 데이터 가져오기",
    "help_fmt_import_body": (
        "파일 메뉴 → '외부 라벨/GT 가져오기...' 를 선택합니다.\n"
        "\n"
        "선택한 폴더 안에 labels/ 또는 gt_image/ 폴더가 있으면\n"
        "현재 프로젝트 폴더로 복사됩니다.\n"
        "\n"
        "외부 폴더 구조 예시:\n"
        "  외부폴더/\n"
        "  ├── labels/\n"
        "  │   └── image1.txt\n"
        "  └── gt_image/\n"
        "      └── cat/\n"
        "          └── image1.png\n"
        "\n"
        "가져오기 후 현재 이미지 라벨이 자동으로 갱신됩니다."
    ),

    "help_fmt_resume_title": "🔄 이전 작업 이어하기",
    "help_fmt_resume_body": (
        "이미지 폴더를 열면 labels/, gt_image/ 데이터를\n"
        "자동으로 감지하고 이미지 목록에 ✔ 표시로 알려줍니다.\n"
        "\n"
        "• 이미지 선택 시 라벨/마스크가 자동 로드됩니다.\n"
        "• Segmentation 모드에서는 마스크가 브러시에 즉시 로드되어\n"
        "  바로 수정 가능합니다.\n"
        "• 새 클래스는 라벨 파일에서 자동 등록됩니다."
    ),
}
