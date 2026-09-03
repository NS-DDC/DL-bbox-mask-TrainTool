"""Real backend integration on GitHub Actions only, never a local PC probe.

    Built-in YAML -> random model -> custom best.pt -> AUTO loader -> CPU
    inference on a synthetic rectangle. No pretrained weights are downloaded.

These tests check the pinned dependency APIs and original-coordinate result
contract, not the accuracy of any trained model or user's dataset. Each backend
runs in a separate process with a 180-second limit and two CPU threads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS", "").lower() != "true",
    reason="Actual model execution is authorized only on GitHub Actions for this delivery.",
)


@pytest.mark.parametrize("backend", ["yolo-segment", "rtdetr"])
def test_real_random_checkpoint_load_and_cpu_prediction(backend, tmp_path):
    environment = os.environ.copy()
    environment.update({
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
        "QT_QPA_PLATFORM": "offscreen",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONUTF8": "1",
    })
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), backend, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    report = json.loads((tmp_path / "backend-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["backend"] == backend
    assert report["device"] == "cpu"
    assert report["original_shape"] == [73, 121]
    assert report["weights"] == "random initialization; no pretrained download"


def _run_probe(backend: str, output_dir: Path) -> None:
    """Executed only in a fresh GitHub runner child process."""
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        raise RuntimeError("Refusing real model execution outside GitHub Actions.")
    if backend not in {"yolo-segment", "rtdetr"}:
        raise ValueError(f"Unknown backend probe: {backend}")

    # Fail closed if a dependency attempts any network connection, including
    # fetching pretrained models, assets or missing packages.
    import socket

    def refuse_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden during random-backend integration tests.")

    socket.socket.connect = refuse_network
    socket.socket.connect_ex = refuse_network
    socket.create_connection = refuse_network
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import gc
    import time

    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO, RTDETR

    from core.image_io import atomic_write_image
    from core.model_manager import ModelManager

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(7)
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "yolo-segment":
        architecture = YOLO("yolo11n-seg.yaml")
        expected_type, expected_task, infer_size = "YOLO", "segment", 64
        expected_predictor = "SegmentationPredictor"
    else:
        architecture = RTDETR("rtdetr-l.yaml")
        expected_type, expected_task, infer_size = "RT-DETR", "detect", 128
        expected_predictor = "RTDETRPredictor"
        # At 128px, RT-DETR has 16²+8²+4²=336 encoder positions for its
        # 300 queries. 64px only has 84 and is not a valid smoke input.

    checkpoint = output_dir / "best.pt"
    architecture.save(str(checkpoint))
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0
    del architecture
    gc.collect()

    original = np.zeros((73, 121, 3), dtype=np.uint8)
    cv2.rectangle(original, (17, 11), (89, 58), (50, 170, 220), thickness=-1)
    image_path = output_dir / "원본 synthetic.png"
    atomic_write_image(image_path, original)

    manager = ModelManager()
    assert manager.load_model(str(checkpoint), "AUTO"), manager.last_error
    assert manager.get_model_type() == expected_type
    assert manager.get_model_task() == expected_task
    assert manager.get_model_path() == str(checkpoint.resolve())
    results = manager.predict(str(image_path), confidence=1.0,
                              infer_size=infer_size, device="cpu")
    assert results is not None, manager.last_error
    assert len(results) == 1
    result = results[0]
    assert tuple(result.orig_shape) == original.shape[:2]
    np.testing.assert_array_equal(result.orig_img, original)
    assert result.boxes is not None
    assert result.boxes.data.device.type == "cpu"
    assert manager.last_device == "cpu"
    assert type(manager.get_model().predictor).__name__ == expected_predictor

    report = {
        "status": "passed",
        "backend": backend,
        "model_type": manager.get_model_type(),
        "task": manager.get_model_task(),
        "device": manager.last_device,
        "infer_size": infer_size,
        "original_shape": list(result.orig_shape),
        "weights": "random initialization; no pretrained download",
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitation": "API and source-coordinate integration only; no trained-model accuracy validation.",
    }
    (output_dir / "backend-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manager.unload()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Expected backend name and temporary output directory.")
    _run_probe(sys.argv[1], Path(sys.argv[2]))
