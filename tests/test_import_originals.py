"""CI-only checks for non-overwriting imports and stable external class IDs."""

from pathlib import Path

import numpy as np
import pytest

from core.image_io import atomic_write_image
from core.import_manager import import_annotations
from core.project_manager import ProjectManager
from core.project_metadata import load_classes


@pytest.fixture
def project_and_external(tmp_path):
    image_dir = tmp_path / "project"
    image_dir.mkdir()
    atomic_write_image(image_dir / "wafer.png", np.zeros((20, 30, 3), np.uint8))
    project = ProjectManager()
    project.open_folder(str(image_dir))
    custom = tmp_path / "custom-labels"
    project.set_custom_label_dir(str(custom))
    external = tmp_path / "external"
    (external / "labels").mkdir(parents=True)
    (external / "labels" / "classes.txt").write_text("defect\n", encoding="utf-8")
    (external / "labels" / "wafer.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return project, external, custom


def test_import_honors_custom_label_dir_and_class_mapping(project_and_external):
    project, external, custom = project_and_external
    counts = import_annotations(external, project, [])
    assert counts[:2] == (1, 0)
    assert (custom / "wafer.txt").read_bytes() == (external / "labels" / "wafer.txt").read_bytes()
    assert not (project.image_dir / "labels" / "wafer.txt").exists()
    assert load_classes(custom)[0]["name"] == "defect"


def test_conflict_does_not_overwrite_files_or_mapping(project_and_external):
    project, external, custom = project_and_external
    original = custom / "wafer.txt"
    original.write_text("0 0.1 0.1 0.1 0.1\n")
    before = original.read_bytes()
    with pytest.raises(FileExistsError, match="Existing annotation differs"):
        import_annotations(external, project, [])
    assert original.read_bytes() == before
    assert not (custom / ".visionace-project.json").exists()


def test_mapping_id_conflict_fails_before_copy(project_and_external):
    project, external, custom = project_and_external
    with pytest.raises(ValueError, match="Class ID 0"):
        import_annotations(external, project, [{"name": "different", "color": "#ffffff"}])
    assert not (custom / "wafer.txt").exists()


def test_gt_dimensions_checked_before_any_labels_are_copied(project_and_external):
    project, external, custom = project_and_external
    atomic_write_image(external / "gt_image" / "defect" / "wafer.png", np.zeros((10, 15), np.uint8))
    with pytest.raises(ValueError, match="dimensions"):
        import_annotations(external, project, [])
    assert not (custom / "wafer.txt").exists()
    assert not (project.image_dir / "gt_image").exists()
