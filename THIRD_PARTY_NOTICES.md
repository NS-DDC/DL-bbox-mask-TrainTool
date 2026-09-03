# Source and dependency notices

The upstream VisionAce README identifies its application source as MIT-licensed. The source baseline and modification scope are recorded in [docs/UPSTREAM.md](docs/UPSTREAM.md). No third-party model weights are included in this release.

Bundled packages keep their own licenses. In particular, the [Ultralytics distribution](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) includes AGPL-3.0 terms; the upstream application's MIT statement does not replace dependency terms. Qt/PySide6, PyTorch, torchvision, OpenCV, NumPy, Pillow and the other installed packages retain their respective notices.

The portable build copies available license/notice files to `third-party-licenses/`, records installed distributions in its `inventory.json`, and includes `requirements-resolved.txt`. Source for this modified application is published alongside the tagged release. Dependency source projects:

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision)
- [Qt for Python](https://code.qt.io/cgit/pyside/pyside-setup.git/)
- [OpenCV](https://github.com/opencv/opencv)
- [NumPy](https://github.com/numpy/numpy)
- [Pillow](https://github.com/python-pillow/Pillow)

Keras/TensorFlow and DINOv3 are not bundled in the CPU portable release.
