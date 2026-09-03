"""Coordinate/conversion and worker lifecycle regressions using synthetic results."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.auto_labeler import AutoLabelWorker, _restore_instance_mask


class FakeBoxes:
    def __init__(self, xyxy=(), classes=(), scores=()):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.cls = np.asarray(classes, np.float32)
        self.conf = np.asarray(scores, np.float32)

    def __len__(self):
        return len(self.conf)


class FakeMasks:
    def __init__(self, polygons, data=None):
        self.xy = polygons
        self.data = data

    def __len__(self):
        return len(self.xy)


class FakeManager:
    is_loaded = True
    last_error = "mock inference failed: invalid checkpoint"

    def __init__(self, results, task="detect"):
        self.results = results
        self.task = task

    def get_model_task(self):
        return self.task

    def get_class_names(self):
        return {0: "scratch", 1: "particle"}

    def predict(self, *args, **kwargs):
        return self.results


def make_result(boxes=None, masks=None, shape=(1000, 2000)):
    return SimpleNamespace(orig_shape=shape, boxes=boxes if boxes is not None else FakeBoxes(),
                           masks=masks, probs=None, keypoints=None, obb=None)


def segment_result():
    polygon = np.asarray([[200, 100], [700, 100], [700, 400], [200, 400]], np.float32)
    boxes = FakeBoxes([[200, 100, 700, 400]], [1], [0.9])
    return make_result(boxes, FakeMasks([polygon]))


def test_segmentation_auto_outputs_one_original_coordinate_polygon_not_duplicate_bbox():
    worker = AutoLabelWorker(FakeManager([segment_result()], "segment"), ["image.png"])
    labels = worker._process_image("image.png", {1: "particle"})
    assert len(labels) == 1
    assert labels[0].label_type == "polygon"
    assert set(labels[0].points) == {(200.0, 100.0), (700.0, 100.0), (700.0, 400.0), (200.0, 400.0)}
    assert labels[0].class_id == 1 and labels[0].class_name == "particle"


def test_explicit_bbox_output_does_not_also_create_mask():
    worker = AutoLabelWorker(FakeManager([segment_result()], "segment"), ["image.png"], output_type="bbox")
    labels = worker._process_image("image.png", {1: "particle"})
    assert [label.label_type for label in labels] == ["bbox"]


def test_bbox_coordinates_clipped_to_original_image_without_rescaling():
    boxes = FakeBoxes([[-5, 100, 2200, 800]], [0], [0.8])
    worker = AutoLabelWorker(FakeManager([make_result(boxes)]), ["image.png"])
    labels = worker._process_image("image.png", {0: "scratch"})
    assert labels[0].points == [(0.0, 100.0), (2000.0, 100.0), (2000.0, 800.0), (0.0, 800.0)]


def test_score_filter_retains_valid_custom_model_detection_below_old_half_default():
    boxes = FakeBoxes([[20, 10, 70, 40], [100, 100, 140, 140]], [0, 1], [0.30, 0.15])
    worker = AutoLabelWorker(FakeManager([make_result(boxes)]), ["image.png"])
    labels = worker._process_image("image.png", {})
    assert len(labels) == 1 and labels[0].class_id == 0


@pytest.mark.parametrize("output", ["polygon", "mask"])
def test_detector_cannot_fabricate_segmentation(output):
    boxes = FakeBoxes([[20, 10, 70, 40]], [0], [0.8])
    worker = AutoLabelWorker(FakeManager([make_result(boxes)]), ["image.png"], output_type=output)
    with pytest.raises(ValueError, match="only returns bboxes"):
        worker._process_image("image.png", {})


def test_failed_inference_never_emits_empty_success_and_reports_finish():
    worker = AutoLabelWorker(FakeManager(None), ["a.png", "b.png"])
    errors, images, completed = [], [], []
    worker.error.connect(errors.append)
    worker.image_done.connect(lambda path, labels: images.append((path, labels)))
    worker.finished_all.connect(lambda: completed.append(True))
    worker.run()
    assert images == []
    assert len(errors) == 2 and all("invalid checkpoint" in message for message in errors)
    assert worker.summary["failed"] == 2 and worker.summary["successful"] == 0
    assert worker.summary["processed"] == 2
    assert completed == [True]


def test_zero_detections_is_a_distinct_successful_empty_result():
    worker = AutoLabelWorker(FakeManager([make_result()]), ["empty.png"])
    images, errors = [], []
    worker.image_done.connect(lambda path, labels: images.append((path, labels)))
    worker.error.connect(errors.append)
    worker.run()
    assert images == [("empty.png", [])]
    assert errors == []
    assert worker.summary["empty"] == 1 and worker.summary["successful"] == 1


def test_missing_results_container_is_failure_not_zero_detections():
    worker = AutoLabelWorker(FakeManager([]), ["image.png"])
    with pytest.raises(ValueError, match="no Results object"):
        worker._process_image("image.png", {})


def test_no_loaded_model_still_emits_completion_signal():
    manager = FakeManager(None)
    manager.is_loaded = False
    worker = AutoLabelWorker(manager, ["a.png"])
    completed, errors = [], []
    worker.finished_all.connect(lambda: completed.append(True))
    worker.error.connect(errors.append)
    worker.run()
    assert completed == [True] and errors == ["No model is loaded."]
    assert worker.summary["fatal_error"] == "No model is loaded."


def test_cancellation_discards_result_finishing_after_abort():
    manager = FakeManager([segment_result()], "segment")
    worker = AutoLabelWorker(manager, ["a.png", "b.png"])
    def predict(*args, **kwargs):
        worker.abort()
        return manager.results
    manager.predict = predict
    images = []
    worker.image_done.connect(lambda path, labels: images.append((path, labels)))
    worker.run()
    assert images == [] and worker.summary["cancelled"] is True
    assert worker.summary["processed"] == 0


def test_non_finite_detections_are_rejected():
    boxes = FakeBoxes([[20, 10, float("nan"), 40]], [0], [0.8])
    worker = AutoLabelWorker(FakeManager([make_result(boxes)]), ["image.png"])
    with pytest.raises(ValueError, match="non-finite"):
        worker._process_image("image.png", {})


def test_mask_unpads_before_resize_on_wide_original():
    # 8x8 network output: original image region occupies rows 2..5.
    padded = np.zeros((8, 8), np.float32)
    padded[2:6, 2:6] = 1
    restored = _restore_instance_mask(padded, (4, 8))
    expected = np.zeros((4, 8), np.uint8)
    expected[:, 2:6] = 255
    np.testing.assert_array_equal(restored, expected)


def test_mask_unpads_before_resize_on_tall_original():
    padded = np.zeros((8, 8), np.float32)
    padded[2:6, 2:6] = 1
    restored = _restore_instance_mask(padded, (8, 4))
    expected = np.zeros((8, 4), np.uint8)
    expected[2:6, :] = 255
    np.testing.assert_array_equal(restored, expected)


def test_native_resolution_mask_preserves_holes_and_disconnected_regions():
    source = np.zeros((15, 30), np.uint8)
    source[2:12, 2:12] = 1
    source[5:8, 5:8] = 0
    source[2:5, 20:25] = 1
    result = make_result(FakeBoxes([[2, 2, 25, 12]], [0], [0.9]),
                         FakeMasks([np.empty((0, 2))], data=source[None]), (15, 30))
    worker = AutoLabelWorker(FakeManager([result], "segment"), ["image.png"], output_type="mask")
    labels = worker._process_image("image.png", {})
    assert len(labels) == 1 and labels[0].label_type == "mask"
    np.testing.assert_array_equal(labels[0].mask_data, source * 255)


@pytest.mark.parametrize("output", ["bbox", "polygon"])
def test_keras_semantic_components_are_not_silently_discarded(output):
    probabilities = np.zeros((1, 20, 40, 1), np.float32)
    probabilities[0, 2:8, 2:8, 0] = 0.9
    probabilities[0, 12:18, 30:36, 0] = 0.9
    worker = AutoLabelWorker(FakeManager(None, "segment"), [], output_type=output)
    labels = worker._process_keras_results({"predictions": probabilities, "orig_shape": (20, 40)}, {})
    assert len(labels) == 2
    assert all(label.label_type == output for label in labels)


def test_keras_rejects_logits_without_guessed_activation():
    worker = AutoLabelWorker(FakeManager(None, "segment"), [])
    with pytest.raises(ValueError, match="probabilities"):
        worker._process_keras_results({"predictions": np.full((1, 10, 10, 1), 3.0), "orig_shape": (20, 40)}, {})


def test_keras_classification_never_becomes_whole_image_bbox():
    worker = AutoLabelWorker(FakeManager(None, "segment"), [])
    with pytest.raises(ValueError, match="classification cannot localize"):
        worker._process_keras_results({"predictions": np.asarray([[0.1, 0.9]]), "orig_shape": (20, 40)}, {})
