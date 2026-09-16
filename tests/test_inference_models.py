"""Mocked model regressions for CI; no weight downloads or real inference."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import core.model_manager as manager_module
from core.model_manager import ModelManager


class DetectHead:
    pass


class RTDETRDecoder:
    pass


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"mock checkpoint; never deserialized")
    return path


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(head=DetectHead, task="detect", names={0: "scratch"},
                            failure=None, loaded=[], predicted=[], result=[], cuda=False)

    class FakeYOLO:
        def __init__(self, path):
            state.loaded.append(("YOLO", path))
            if state.failure:
                raise state.failure
            self.model = SimpleNamespace(model=[state.head()])
            self.task = state.task
            self.names = state.names
            self.predictor = object()

        def predict(self, **kwargs):
            state.predicted.append(kwargs)
            if state.failure:
                raise state.failure
            return state.result

    class FakeRTDETR(FakeYOLO):
        def __init__(self, path):
            super().__init__(path)
            state.loaded[-1] = ("RT-DETR", path)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO, RTDETR=FakeRTDETR))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: state.cuda, device_count=lambda: 1 if state.cuda else 0)))
    return state


def test_missing_local_path_never_invokes_download_capable_backend(tmp_path, backend):
    manager = ModelManager()
    assert not manager.load_model(str(tmp_path / "yolo11n.pt"))
    assert "does not exist" in manager.last_error
    assert backend.loaded == []


def test_auto_detects_custom_named_rtdetr_checkpoint(checkpoint, backend):
    backend.head = RTDETRDecoder
    manager = ModelManager()
    assert manager.load_model(str(checkpoint), "AUTO")
    assert manager.get_model_type() == "RT-DETR"
    assert manager.get_model_task() == "detect"
    assert [kind for kind, _ in backend.loaded] == ["YOLO", "RT-DETR"]
    assert all(path == str(checkpoint.resolve()) for _, path in backend.loaded)


def test_explicit_rtdetr_rejects_yolo_head(checkpoint, backend):
    manager = ModelManager()
    assert not manager.load_model(str(checkpoint), "RT-DETR")
    assert "different architecture" in manager.last_error


@pytest.mark.parametrize("problem", ["load_exception", "classification", "dino"])
def test_failed_replacement_preserves_working_model(checkpoint, backend, problem):
    manager = ModelManager()
    assert manager.load_model(str(checkpoint))
    previous = manager.get_model()
    if problem == "load_exception":
        backend.failure = ValueError("checkpoint shape mismatch")
    elif problem == "classification":
        backend.task = "classify"
    assert not manager.load_model(str(checkpoint), "DINOv3" if problem == "dino" else "AUTO")
    assert manager.get_model() is previous
    assert manager.get_model_type() == "YOLO"
    assert manager.get_model_path() == str(checkpoint.resolve())
    assert manager.last_error


def test_class_names_allow_list_as_well_as_dictionary(checkpoint, backend):
    backend.names = ["scratch", "particle"]
    manager = ModelManager()
    assert manager.load_model(str(checkpoint))
    assert manager.get_class_names() == {0: "scratch", 1: "particle"}


def test_original_decoded_pixels_and_cpu_are_passed_to_backend(checkpoint, backend, monkeypatch):
    original = np.zeros((713, 1931, 3), np.uint8)
    monkeypatch.setattr(manager_module, "read_image", lambda path: original)
    manager = ModelManager()
    assert manager.load_model(str(checkpoint))
    assert manager.predict("원본/웨이퍼.png", device="auto") == []
    call = backend.predicted[-1]
    assert call["source"] is original
    assert call["source"].shape == (713, 1931, 3)
    assert call["device"] == "cpu" and call["half"] is False
    assert call["save"] is False
    assert manager.last_device == "cpu"


def test_segmentation_requests_original_size_raster_masks(checkpoint, backend, monkeypatch):
    backend.task = "segment"
    monkeypatch.setattr(manager_module, "read_image", lambda path: np.zeros((40, 90, 3), np.uint8))
    manager = ModelManager()
    assert manager.load_model(str(checkpoint))
    manager.predict("image.png", device="cpu")
    assert backend.predicted[-1]["retina_masks"] is True


def test_unavailable_explicit_gpu_is_actionable_and_never_silent(checkpoint, backend, monkeypatch):
    monkeypatch.setattr(manager_module, "read_image", lambda path: np.zeros((10, 20, 3), np.uint8))
    manager = ModelManager()
    assert manager.load_model(str(checkpoint))
    assert manager.predict("image.png", device="cuda:0") is None
    assert "unavailable" in manager.last_error and "cpu" in manager.last_error
    assert backend.predicted == []


def test_device_change_discards_cached_predictor(checkpoint, backend, monkeypatch):
    backend.cuda = True
    monkeypatch.setattr(manager_module, "read_image", lambda path: np.zeros((10, 20, 3), np.uint8))
    manager = ModelManager()
    assert manager.load_model(str(checkpoint))
    manager.predict("image.png", device="0")
    sentinel = object()
    manager.get_model().predictor = sentinel
    manager.predict("image.png", device="cpu")
    assert manager.get_model().predictor is None
    assert [call["device"] for call in backend.predicted] == ["0", "cpu"]


def test_decode_failure_does_not_infer(checkpoint, backend, monkeypatch):
    monkeypatch.setattr(manager_module, "read_image", lambda path: None)
    manager = ModelManager()
    assert manager.load_model(str(checkpoint))
    assert manager.predict("broken.tif") is None
    assert "Cannot decode image" in manager.last_error
    assert backend.predicted == []


def test_out_of_memory_retains_original_error_and_recovery_hint(checkpoint, backend, monkeypatch):
    monkeypatch.setattr(manager_module, "read_image", lambda path: np.zeros((10, 20, 3), np.uint8))
    manager = ModelManager()
    assert manager.load_model(str(checkpoint))
    backend.failure = RuntimeError("CUDA out of memory")
    assert manager.predict("image.png") is None
    assert "CUDA out of memory" in manager.last_error
    assert "Reduce inference size" in manager.last_error


def test_dinov3_backbone_is_not_mislabeled_as_detector(checkpoint, backend):
    manager = ModelManager()
    assert not manager.load_model(str(checkpoint), "DINOv3")
    assert "backbone weights alone" in manager.last_error
    assert "head" in manager.last_error
    assert backend.loaded == []


def test_keras_classification_is_not_accepted_for_bboxes(tmp_path, monkeypatch):
    path = tmp_path / "classifier.keras"
    path.write_bytes(b"mock")
    keras = ModuleType("keras")
    models = ModuleType("keras.models")
    models.load_model = lambda *a, **k: SimpleNamespace(input_shape=(None, 32, 32, 3), output_shape=(None, 4))
    keras.models = models
    monkeypatch.setitem(sys.modules, "keras", keras)
    monkeypatch.setitem(sys.modules, "keras.models", models)
    manager = ModelManager()
    assert not manager.load_model(str(path))
    assert "Classification scores do not localize objects" in manager.last_error
