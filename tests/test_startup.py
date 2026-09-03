"""Startup contract checks, executed by GitHub Actions only for this delivery."""
import importlib
import json
import logging

import pytest


def test_import_does_not_start_qt_or_load_models():
    # main keeps the expensive and GUI imports inside main(), not module scope.
    module = importlib.import_module("main")
    assert callable(module.main)
    assert module.APP_VERSION == "1.9.0-improved.3"


def test_version_does_not_need_gui(monkeypatch, capsys):
    import main
    monkeypatch.setattr(main, "configure_logging", lambda: pytest.fail("Version must not initialize GUI logging"))
    with pytest.raises(SystemExit) as result:
        main.main(["--version"])
    assert result.value.code == 0
    assert main.APP_VERSION in capsys.readouterr().out


def test_logging_uses_configured_writable_directory(tmp_path, monkeypatch):
    import main
    monkeypatch.setenv("VISIONACE_HOME", str(tmp_path))
    previous_handlers = set(logging.getLogger().handlers)
    previous_level = logging.getLogger().level
    previous_warning_hook = __import__("warnings").showwarning
    try:
        path = main.configure_logging()
        logging.getLogger("startup-test").warning("diagnostic-marker")
        assert path == tmp_path / "logs" / "visionace.log"
        assert "diagnostic-marker" in path.read_text(encoding="utf-8")
    finally:
        for handler in list(logging.getLogger().handlers):
            if handler not in previous_handlers:
                logging.getLogger().removeHandler(handler)
                handler.close()
        logging.getLogger().setLevel(previous_level)
        __import__("warnings").showwarning = previous_warning_hook


def test_smoke_report_can_be_written_under_unicode_directory(tmp_path):
    import main
    target = tmp_path / "검증 결과" / "smoke.json"
    main._write_report(target, {"status": "failed", "error": "모델 없음"})
    assert json.loads(target.read_text(encoding="utf-8"))["error"] == "모델 없음"


def test_bundled_font_covers_printable_ascii_and_all_modern_hangul():
    import main
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    previous_font = app.font()
    try:
        family = main.configure_application_font(app)
        report = main._verify_application_font(app)
        assert report["registered_family"] == family
        assert report["missing_glyphs"] == 0
        assert report["ascii_glyphs_checked"] == 95
        assert report["hangul_syllables_checked"] == 11172
    finally:
        app.setFont(previous_font)


def test_missing_bundled_font_fails_startup(tmp_path):
    import main
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    with pytest.raises(RuntimeError, match="Cannot load the bundled application font"):
        main.configure_application_font(app, tmp_path / "missing-font.ttf")


def test_missing_glyph_fails_the_smoke_gate(monkeypatch):
    import main
    from PySide6 import QtGui
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    class MissingHangulMetrics:
        def __init__(self, font):
            pass

        def inFontUcs4(self, point):
            return point != 0xAC00

    monkeypatch.setattr(QtGui, "QFontMetrics", MissingHangulMetrics)
    with pytest.raises(RuntimeError, match="U\\+AC00"):
        main._verify_application_font(app)


def test_font_loads_from_unicode_frozen_resource_root(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    import main
    from PySide6 import QtGui
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    source = Path(main.__file__).resolve().parent / "assets" / "fonts" / "NotoSansKR.ttf"
    frozen_root = tmp_path / "압축 해제 검증" / "_internal"
    bundled = frozen_root / "assets" / "fonts" / "NotoSansKR.ttf"
    bundled.parent.mkdir(parents=True)
    font_bytes = source.read_bytes()
    bundled.write_bytes(font_bytes)
    monkeypatch.setattr(main.sys, "_MEIPASS", str(frozen_root), raising=False)
    monkeypatch.setattr(QtGui.QFontDatabase, "addApplicationFont",
                        lambda path: pytest.fail("Qt must receive font bytes, not a filename"))
    previous_font = app.font()
    try:
        main.configure_application_font(app)
        report = main._verify_application_font(app)
        assert report["font_sha256"] == hashlib.sha256(font_bytes).hexdigest()
        assert report["font_bytes"] == len(font_bytes)
        assert report["missing_glyphs"] == 0
    finally:
        app.setFont(previous_font)


def test_invalid_font_data_still_fails_startup(tmp_path):
    import main
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    invalid = tmp_path / "invalid.ttf"
    invalid.write_bytes(b"not a valid TrueType font")
    with pytest.raises(RuntimeError, match="Cannot load the bundled application font from data"):
        main.configure_application_font(app, invalid)
