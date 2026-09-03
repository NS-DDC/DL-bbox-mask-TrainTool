"""English translations for VisionAce."""

EN = {
    # Application
    "app_title": "VisionAce - Deep Learning Labeling & Training Tool  |  by KNS",

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
    "label_show_all": "Show All",
    "label_hide_all": "Hide All",

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
    "auto_label_progress_pct": "{current} / {total}  ({pct}%)",
    "auto_label_time_info": "Speed: {speed} img/s  |  ETA: {eta}  |  Elapsed: {elapsed}",
    "auto_label_elapsed": "Total time: {elapsed}",
    "auto_label_complete": "Auto labeling complete: {count} labels generated",
    "auto_label_no_model": "No model loaded. Please load a model first.",
    "auto_label_score_threshold": "Score Threshold:",
    "auto_label_score_threshold_tooltip": (
        "Post-processing confidence filter. Detections with a score below this "
        "value are discarded after inference. Set higher than Confidence for stricter filtering."
    ),
    "auto_label_infer_size": "Inference Size (px):",
    "auto_label_infer_size_tooltip": (
        "Square image size used for YOLO / RT-DETR inference (e.g. 480). "
        "Coordinates are automatically back-projected to original image resolution."
    ),
    "auto_label_confidence_tooltip": (
        "Minimum confidence passed to the model (NMS threshold). "
        "Use a low value to catch more detections, then filter with Score Threshold."
    ),
    "auto_label_device": "Device:",
    "auto_label_device_tooltip": (
        "Device for inference (YOLO/RT-DETR only). "
        "auto: use GPU if available, "
        "cpu: force CPU, "
        "0: GPU 0. You can also type a custom value (e.g. 1, mps)."
    ),

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
    "help_shortcut_save": "Ctrl+S : Save labels (GT masks always saved as PNG)",
    "help_shortcut_undo": "Ctrl+Z : Undo",
    "help_shortcut_redo": "Ctrl+Y : Redo",
    "help_shortcut_delete": "Delete : Delete selected label",
    "help_shortcut_prev": "A : Previous image (no save)",
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
    "help_tool_segmentation": "Segmentation mode (E): Left-click drag to brush, right-click to erase. Ctrl+click for polygon mode (Enter/right-click to finish). Masks are always saved as PNG.",
    "help_tip_zoom": "Segmentation mode: Mouse wheel adjusts brush size, Ctrl+wheel to zoom. Other modes: Mouse wheel to zoom.",
    "help_tip_pan": "Middle mouse button to pan.",
    "help_tip_brush": "Choose brush shape (circle/square) and size from toolbar.",
    "help_tip_bbox_mode": "Select BBox mode: Rectangle or Polygon.",
    "help_tip_class_color": "Double-click class name to change color.",
    "help_tip_right_click": "Right-click while drawing polygon to finish quickly.",
    "help_tip_autosave": "Auto-save when switching images. Use A (prev), S (save & next), D (skip & next) for quick navigation.",
    "help_tip_save_extension": "GT masks (Segmentation) are always saved as PNG (lossless) to prevent noise artifacts. Original images are copied to images/ folder with their original extension.",
    "help_tip_recent_folders": "Access recently opened folders quickly from File > Recent Folders menu.",
    "help_tip_exclude": "Press F to exclude current image from training. Original, label, and GT images will all be deleted.",
    "help_tip_mask_edit": "Double-click a mask label in SELECT mode to re-edit it with the brush tool.",
    "help_tip_import_labels": "Use File > Import External Labels/GT to import labels and GT from another folder into the current project.",
    "help_tip_label_format": "Label files are saved as 'class_name class_id coords...'. Legacy format (class_id only) is also supported.",
    "help_tip_resume_work": "Opening an image folder automatically loads existing labels/ and gt_image/ data so you can continue editing. In Segmentation mode, selecting an image auto-loads its mask into the brush for immediate editing.",
    "help_shortcut_class_number": "1~9, 0 : Quick class switch (1=first, 0=tenth)",
    "help_shortcut_brush_plus": "+ / Wheel Up (Seg mode) : Increase brush size (+5)",
    "help_shortcut_brush_minus": "- / Wheel Down (Seg mode) : Decrease brush size (-5)",
    "help_shortcut_zoom_in": "Ctrl+Wheel Up / Ctrl++ : Zoom in",
    "help_shortcut_zoom_out": "Ctrl+Wheel Down / Ctrl+- : Zoom out",
    "help_close": "Close",

    # Menu - Help
    "menu_help": "&Help",
    "action_help_dialog": "Help Dialog",
    "action_toggle_help_panel": "Show Help Panel",

    # Help dock
    "help_dock_title": "Help (F1 or ? button)",

    # Recent menus
    "menu_recent_dirs": "Recent Folders",
    "menu_recent_models": "Recent Models",
    "menu_recent_none": "(None)",
    "menu_recent_clear": "Clear List",

    # Navigation actions
    "action_prev_image": "Previous Image (A)",
    "action_next_save": "Save & Next Image (S)",
    "action_next_no_save": "Next Image without Saving (D)",
    "action_exclude_training": "Exclude from Training (F)",

    # Settings
    "action_set_save_extension": "Set Save Image Extension...",
    "help_tooltip": "Open/Close Help (F1)",

    # Save extension
    "save_ext_label": "Save Ext:",
    "save_ext_original": "Original",
    "save_ext_tooltip": "Image save extension",

    # Navigation buttons
    "nav_prev": "◀ Prev [A]",
    "nav_prev_tooltip": "Go to previous image (Shortcut: A)",
    "nav_next_save": "Save+Next [S]",
    "nav_next_save_tooltip": "Save and go to next image (Shortcut: S)",
    "nav_next_no_save": "Skip+Next [D]",
    "nav_next_no_save_tooltip": "Go to next image without saving (Shortcut: D)",

    # Toolbar labels
    "toolbar_bbox_mode": "BBox Mode:",
    "toolbar_brush": "Brush:",
    "toolbar_finish": "Finish [Enter]",

    # Status messages
    "status_next_no_save": "Next image (not saved)",
    "status_next_with_save": "Saved and moved to next image",
    "status_prev_image": "Moved to previous image",
    "status_skipped": "Skipped to next image (not saved)",

    # Exclude from training
    "exclude_title": "Exclude from Training",
    "exclude_confirm": "Exclude '{name}' from training?\n\nThe following will be deleted:\n- Original image\n- Label file (.txt)\n- GT image files\n- Copy in images/ folder",
    "exclude_done": "'{name}' excluded from training (deleted)",

    # Folder not found
    "folder_not_found_title": "Folder Not Found",
    "folder_not_found_msg": "Folder does not exist:\n{path}",
    "model_not_found_title": "Model File Not Found",
    "model_not_found_msg": "Model file does not exist:\n{path}",

    # Save extension dialog
    "save_ext_dialog_title": "Set Default Save Image Extension",
    "save_ext_dialog_msg": "Current setting: {current}\n\nSelect image format for auto-save:\n(Ctrl+S full save allows separate selection)",
    "save_ext_use_original": "Use Original Extension",
    "save_ext_status": "Default save extension: {ext}",
    "save_ext_display_original": "Original Extension",
    "save_ext_toolbar_status": "Save extension: {ext}",

    # Import external labels
    "action_import_labels": "Import External Labels/GT...",
    "import_select_folder": "Select External Labels Folder",
    "import_no_project": "Please open an image folder first.",
    "import_no_data": "No labels/ or gt_image/ folder found in selected folder.",
    "import_complete": "Import complete: {labels} labels, {gt} GT images copied",

    # Mask edit
    "mask_edit_status": "Mask edit mode - Modify with brush, Enter to finish",

    # Export mask format
    "export_mask_format_title": "Mask Format Selection",
    "export_mask_format_message": "Export as multi-label semantic mask?\nYes: Pixel value = Class ID + 1 (semantic segmentation)\nNo: Binary mask (foreground=255, background=0)",

    # Help – Data formats section
    "help_formats_title": "Data Formats & Import",

    "help_fmt_folder_title": "📁 Folder Structure",
    "help_fmt_folder_body": (
        "When you open an image folder, the following layout is auto-detected:\n"
        "\n"
        "  <image_folder>/\n"
        "  ├── image1.jpg       ← source images\n"
        "  ├── image2.png\n"
        "  ├── labels/          ← BBox/Polygon YOLO labels\n"
        "  │   ├── image1.txt\n"
        "  │   └── image2.txt\n"
        "  ├── gt_image/        ← Segmentation GT masks\n"
        "  │   ├── classA/\n"
        "  │   │   ├── image1.png\n"
        "  │   │   └── image2.png\n"
        "  │   └── classB/\n"
        "  │       └── image1.png\n"
        "  └── images/          ← images copied on save"
    ),

    "help_fmt_yolo_title": "📄 YOLO Label Format (labels/*.txt)",
    "help_fmt_yolo_body": (
        "BBox:     <class_name> <class_id> <cx> <cy> <w> <h>  (all 0-1 normalised)\n"
        "Example:  cat 0 0.5 0.4 0.2 0.3\n"
        "\n"
        "Polygon:  <class_name> <class_id> <x1> <y1> <x2> <y2> ...\n"
        "Example:  dog 1 0.1 0.2 0.3 0.4 0.5 0.2\n"
        "\n"
        "Note: legacy format (class_id only as first column) is also supported."
    ),

    "help_fmt_gtmask_title": "🖼️ GT Mask Format (gt_image/<class_name>/)",
    "help_fmt_gtmask_body": (
        "• Binary PNG image (foreground=255, background=0)\n"
        "• File stem must match the source image (extension excluded)\n"
        "• Stored per class in gt_image/cat/image1.png\n"
        "• Always saved as PNG — prevents JPEG compression noise\n"
        "• On load, pixels ≥ 128 are treated as foreground"
    ),

    "help_fmt_import_title": "📥 Importing External Data",
    "help_fmt_import_body": (
        "Go to File → 'Import External Labels/GT...' and select a folder.\n"
        "\n"
        "If the folder contains a labels/ or gt_image/ subfolder,\n"
        "those files are copied into the current project folder.\n"
        "\n"
        "Expected external folder structure:\n"
        "  external_folder/\n"
        "  ├── labels/\n"
        "  │   └── image1.txt\n"
        "  └── gt_image/\n"
        "      └── cat/\n"
        "          └── image1.png\n"
        "\n"
        "Labels are refreshed automatically after import."
    ),

    "help_fmt_resume_title": "🔄 Resuming Previous Work",
    "help_fmt_resume_body": (
        "Opening an image folder auto-detects labels/ and gt_image/ data.\n"
        "Images with existing labels show a ✔ indicator in the file list.\n"
        "\n"
        "• Labels and masks load automatically when you select an image.\n"
        "• In Segmentation mode, the mask is loaded into the brush immediately\n"
        "  so you can start editing right away.\n"
        "• New classes found in label files are registered automatically."
    ),
}
