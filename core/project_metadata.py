"""Versioned, atomic project class mapping; source images are never modified."""

import json
import os
import tempfile
from pathlib import Path


def validate_class_name(name: str) -> str:
    name = name.strip()
    if (not name or name in {".", ".."} or name.endswith((".", " "))
            or any(c in name for c in '<>:"/\\|?*')
            or any(ord(c) < 32 for c in name)):
        raise ValueError("Class names must be valid folder names without /, \\, or reserved characters.")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
    }
    if name.split(".")[0].upper() in reserved:
        raise ValueError("This class name is reserved by Windows.")
    return name


def load_classes(label_dir) -> list[dict]:
    path = Path(label_dir) / ".visionace-project.json"
    if not path.exists():
        # Honor common external YOLO class maps, retaining line-number IDs.
        classes_path = Path(label_dir) / "classes.txt"
        if not classes_path.exists():
            classes_path = Path(label_dir).parent / "classes.txt"
        if not classes_path.exists():
            return []
        names = [validate_class_name(name) for name in
                 classes_path.read_text(encoding="utf-8-sig").splitlines()]
        if len(set(name.casefold() for name in names)) != len(names):
            raise ValueError("External classes.txt contains duplicate class names.")
        return [{"name": name, "color": "#00aaff"} for name in names]
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("classes"), list):
        raise ValueError(f"Unsupported project class mapping: {path}")
    classes = data["classes"]
    names = [validate_class_name(c["name"]) for c in classes]
    if len(set(n.casefold() for n in names)) != len(names):
        raise ValueError("Project has duplicate class names.")
    return [{"name": name, "color": c.get("color", "#00aaff")}
            for name, c in zip(names, classes)]


def save_classes(label_dir, classes: list[dict]) -> None:
    directory = Path(label_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        validate_class_name(cls["name"])
    names = [c["name"].casefold() for c in classes]
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique (case-insensitive).")
    fd, tmp = tempfile.mkstemp(prefix=".classes-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"version": 1, "classes": classes}, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, directory / ".visionace-project.json")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
