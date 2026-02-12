"""Application configuration for VisionAce."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".visionace")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


@dataclass
class AppConfig:
    language: str = "en"
    recent_image_dir: str = ""
    recent_model_path: str = ""
    recent_export_dir: str = ""
    auto_save: bool = True
    bbox_coord_format: str = "absolute"  # "absolute" or "relative"
    default_confidence: float = 0.5
    default_model_type: str = "RT-DETR"
    window_width: int = 1400
    window_height: int = 900
    class_colors: dict = field(default_factory=dict)
    recent_directories: list = field(default_factory=list)  # List of recently opened directories
    recent_models: list = field(default_factory=list)  # List of recently loaded models
    default_save_extension: str = ""  # Empty = use original extension, or ".png", ".jpg", etc.

    def add_recent_directory(self, path: str, max_recent: int = 10):
        """Add a directory to recent directories list (most recent first)."""
        # Remove if already exists
        if path in self.recent_directories:
            self.recent_directories.remove(path)
        # Add to front
        self.recent_directories.insert(0, path)
        # Keep only max_recent items
        self.recent_directories = self.recent_directories[:max_recent]
        self.save()

    def add_recent_model(self, path: str, max_recent: int = 10):
        """Add a model to recent models list (most recent first)."""
        # Remove if already exists
        if path in self.recent_models:
            self.recent_models.remove(path)
        # Add to front
        self.recent_models.insert(0, path)
        # Keep only max_recent items
        self.recent_models = self.recent_models[:max_recent]
        self.save()

    def save(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> "AppConfig":
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except Exception:
                pass
        return cls()


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config
