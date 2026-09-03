"""Startup contract checks, executed by GitHub Actions only for this delivery."""
import importlib
import json
import logging

import pytest


def test_import_does_not_start_qt_or_load_models():
    # main keeps the expensive and GUI imports inside main(), not module scope.
    module = importlib.import_module("main")
    assert callable(module.main)
    assert module.APP_VERSION == "1.9.0-improved.1"


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
