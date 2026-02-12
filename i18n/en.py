"""English translations for VisionAce."""

EN = {
    # Application
    "app_title": "VisionAce - Deep Learning Labeling & Training Tool",

    # Menu - File
    "menu_file": "&File",
    "action_open_folder": "Open Image Folder",
    "action_load_model": "Load Model (.pt, .h5)",
    "action_save_labels": "Save Labels",
    "action_export_masks": "Export Binary Masks",
    "action_exit": "Exit",

    # Menu - Edit
    "menu_edit": "&Edit",
    "action_undo": "Undo",
    "action_redo": "Redo",
    "action_delete_label": "Delete Selected Label",

    # Menu - Tools
    "menu_tools": "&Tools",
    "action_auto_label": "Auto Labeling...",
    "action_training": "Training...",

    # Menu - Settings
    "menu_settings": "&Settings",
    "action_set_label_dir": "Set Label Folder...",
    "action_lang_ko": "Korean (한국어)",
    "action_lang_en": "English",

    # Label folder settings
    "label_dir_title": "Label Folder",
    "label_dir_message": "Do you want to specify a custom label folder?\n\nYes: Choose a custom folder\nNo: Use default labels/ subfolder",
    "label_dir_no_project": "Please open an image folder first.",
    "label_dir_error": "Failed to set label folder.",
    "select_label_folder": "Select Label Folder",

    # Toolbar
    "tool_detection": "Detection (BBox) [W]",
    "tool_segmentation": "Segmentation (Mask) [E]",
    "tool_select": "Select / Move [Q]",

    # File list panel
    "file_panel_title": "Image Files",
    "file_no_folder": "No folder loaded",
    "file_count": "{count} images",

    # Label list panel
    "label_panel_title": "Labels",
    "label_classes_title": "Classes",
    "label_instances_title": "Instances",
    "label_add_class": "Add Class",
    "label_remove_class": "Remove Class",
    "label_class_name": "Class Name:",
    "label_no_labels": "No labels for this image",

    # BBox coordinate format
    "bbox_coord_format": "Coordinates",
    "bbox_coord_absolute": "Absolute (px)",
    "bbox_coord_relative": "Relative (0~1)",
    "bbox_info_absolute": "[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]",
    "bbox_info_relative": "[{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}]",
    "polygon_info_absolute": "{n} pts",
    "polygon_info_relative": "{n} pts (norm)",

    # Canvas
    "canvas_no_image": "Open an image folder to start labeling",
    "canvas_zoom": "Zoom: {zoom}%",

    # Auto labeling dialog
    "auto_label_title": "Auto Labeling",
    "auto_label_model": "Model:",
    "auto_label_confidence": "Confidence Threshold:",
    "auto_label_scope": "Scope:",
    "auto_label_current": "Current Image",
    "auto_label_all": "All Images",
    "auto_label_start": "Start",
    "auto_label_cancel": "Cancel",
    "auto_label_progress": "Processing {current}/{total}...",
    "auto_label_complete": "Auto labeling complete: {count} labels generated",
    "auto_label_no_model": "No model loaded. Please load a model first.",

    # Training dialog
    "training_title": "Model Training",
    "training_model_type": "Model Type:",
    "training_base_model": "Base Model:",
    "training_dataset": "Dataset Path:",
    "training_browse": "Browse...",
    "training_epochs": "Epochs:",
    "training_batch_size": "Batch Size:",
    "training_img_size": "Image Size:",
    "training_lr": "Learning Rate:",
    "training_device": "Device:",
    "training_start": "Start Training",
    "training_stop": "Stop Training",
    "training_close": "Close",
    "training_log": "Training Log",
    "training_progress": "Epoch {epoch}/{total} - Loss: {loss:.4f}",
    "training_complete": "Training complete! Best model saved at: {path}",
    "training_generate_yaml": "Auto-generate data.yaml",
    "training_classes": "Classes:",
    "training_train_path": "Train Path:",
    "training_val_path": "Val Path:",

    # Status bar
    "status_ready": "Ready",
    "status_mode": "Mode: {mode}",
    "status_image_info": "{filename} | {width}x{height}",
    "status_cursor": "({x}, {y})",
    "status_model_loaded": "Model loaded: {name}",
    "status_no_model": "No model loaded",

    # General
    "ok": "OK",
    "cancel": "Cancel",
    "yes": "Yes",
    "no": "No",
    "error": "Error",
    "warning": "Warning",
    "info": "Information",
    "confirm_delete": "Are you sure you want to delete this?",
    "select_folder": "Select Image Folder",
    "select_model": "Select Model File",
    "model_files": "Model Files (*.pt)",

    # Help dialog
    "help_title": "Help - VisionAce",
    "help_workflow_title": "Workflow",
    "help_shortcuts_title": "Keyboard Shortcuts",
    "help_tools_title": "Tool Usage",
    "help_tips_title": "Tips",
    "help_workflow_step1": "1. Open image folder (Ctrl+O)",
    "help_workflow_step2": "2. Add classes and set colors",
    "help_workflow_step3": "3. Select mode: Detection(B) or Segmentation(S)",
    "help_workflow_step4": "4. Perform labeling",
    "help_workflow_step4_bbox": "   • BBox: Drag to draw box",
    "help_workflow_step4_seg": "   • Segmentation: Brush to draw mask",
    "help_workflow_step5": "5. Skip unlabeled images with X key",
    "help_workflow_step6": "6. Save (Ctrl+S) or auto-save",
    "help_workflow_result": "Output folders:",
    "help_workflow_result_images": "   • images/: Labeled original images",
    "help_workflow_result_labels": "   • labels/: BBox/Polygon YOLO labels",
    "help_workflow_result_gtimage": "   • gt_image/: Segmentation masks",
    "help_shortcut_open": "Ctrl+O : Open image folder",
    "help_shortcut_save": "Ctrl+S : Save labels (with toolbar extension)",
    "help_shortcut_undo": "Ctrl+Z : Undo",
    "help_shortcut_redo": "Ctrl+Y : Redo",
    "help_shortcut_delete": "Delete : Delete selected label",
    "help_shortcut_prev": "A : Previous image",
    "help_shortcut_next_save": "S : Save and next image",
    "help_shortcut_next_no_save": "D : Next image without saving",
    "help_shortcut_exclude": "F : Exclude from training (delete image/label/GT)",
    "help_shortcut_quit": "Ctrl+Q : Quit",
    "help_shortcut_select": "Q : Select/Edit mode",
    "help_shortcut_bbox": "W : Detection (BBox) mode",
    "help_shortcut_segmentation": "E : Segmentation (Mask) mode",
    "help_shortcut_enter": "Enter : Finish current polygon/mask",
    "help_shortcut_esc": "Esc : Cancel current drawing",
    "help_shortcut_f1": "F1 : Toggle help panel",
    "help_shortcut_skip": "X : Skip to next image (ignore auto-save)",
    "help_tool_select": "Select mode (V): Click to select labels, drag handles to resize/move.",
    "help_tool_bbox": "Detection mode (B): Rectangle mode - drag to draw. Polygon mode - click to add points, Enter/right-click to finish.",
    "help_tool_segmentation": "Segmentation mode (S): Left-click drag to brush, right-click to erase. Ctrl+click for polygon mode (Enter/right-click to finish).",
    "help_tip_zoom": "Mouse wheel to zoom, Ctrl+wheel to change brush size.",
    "help_tip_pan": "Middle mouse button to pan.",
    "help_tip_brush": "Choose brush shape (circle/square) and size from toolbar.",
    "help_tip_bbox_mode": "Select BBox mode: Rectangle or Polygon.",
    "help_tip_class_color": "Double-click class name to change color.",
    "help_tip_right_click": "Right-click while drawing polygon to finish quickly.",
    "help_tip_autosave": "Auto-save when switching images. Use A (prev), S (save & next), D (skip & next) for quick navigation.",
    "help_tip_save_extension": "Select image save format (Original/PNG/JPG etc.) from 'Save Extension' dropdown in toolbar. Ctrl+S saves with selected format.",
    "help_tip_recent_folders": "Access recently opened folders quickly from File > Recent Folders menu.",
    "help_tip_exclude": "Press F to exclude current image from training. Original, label, and GT images will all be deleted.",
    "help_close": "Close",
}
