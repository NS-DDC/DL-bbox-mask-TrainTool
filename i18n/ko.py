"""Korean translations for VisionAce."""

KO = {
    # Application
    "app_title": "VisionAce - 딥러닝 라벨링 & 학습 도구",

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
    "auto_label_complete": "오토 라벨링 완료: {count}개 라벨 생성",
    "auto_label_no_model": "모델이 로드되지 않았습니다. 먼저 모델을 로드해주세요.",

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
    "help_shortcut_save": "Ctrl+S : 라벨 저장 (툴바 확장자로 저장)",
    "help_shortcut_undo": "Ctrl+Z : 실행 취소",
    "help_shortcut_redo": "Ctrl+Y : 다시 실행",
    "help_shortcut_delete": "Delete : 선택 라벨 삭제",
    "help_shortcut_prev": "A : 이전 이미지",
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
    "help_tool_segmentation": "Segmentation 모드 (S): 좌클릭 드래그로 브러시 그리기, 우클릭으로 지우기. Ctrl+클릭으로 폴리곤 그리기 (Enter/우클릭으로 완료).",
    "help_tip_zoom": "마우스 휠로 확대/축소, Ctrl+휠로 브러시 크기 조절.",
    "help_tip_pan": "마우스 중간 버튼(휠 클릭)으로 화면 이동.",
    "help_tip_brush": "브러시 모양(원형/사각형)과 크기를 툴바에서 선택 가능.",
    "help_tip_bbox_mode": "BBox 모드를 Rectangle/Polygon 중 선택 가능.",
    "help_tip_class_color": "클래스 목록에서 더블클릭하면 색상 변경.",
    "help_tip_autosave": "다른 이미지로 이동하면 자동 저장됩니다. A(이전), S(저장O 다음), D(저장X 다음) 키로 빠른 이동.",
    "help_tip_right_click": "폴리곤 그리는 중 우클릭으로 빠르게 완료.",
    "help_tip_save_extension": "툴바의 '저장 확장자' 드롭다운에서 이미지 저장 형식(원본/PNG/JPG 등)을 선택하세요. Ctrl+S 저장 시 선택한 형식으로 저장됩니다.",
    "help_tip_recent_folders": "파일 > 최근 폴더 메뉴에서 최근 작업한 폴더를 빠르게 열 수 있습니다.",
    "help_tip_exclude": "F키로 현재 이미지를 학습에서 제외할 수 있습니다. 원본, 라벨, GT 이미지가 모두 삭제됩니다.",
    "help_close": "닫기",
}
