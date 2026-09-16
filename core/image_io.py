"""Unicode-safe image I/O without changing source pixels or EXIF geometry.

Display and inference must both use ``read_image``: OpenCV and Qt otherwise
disagree about EXIF orientation and supported TIFF formats on some platforms.
"""

from __future__ import annotations

import filecmp
import logging
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode at the stored resolution, ignoring EXIF display rotation.

    The default produces an 8-bit BGR display/inference buffer; it never writes
    to the source. Use IMREAD_UNCHANGED when the original bit depth is needed.
    Reading bytes first avoids OpenCV's Windows non-ASCII filename limitation.
    """
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        if not encoded.size:
            return None
        decode_flags = flags if flags == cv2.IMREAD_UNCHANGED else flags | cv2.IMREAD_IGNORE_ORIENTATION
        return cv2.imdecode(encoded, decode_flags)
    except (OSError, ValueError, cv2.error):
        logger.exception("Could not decode image: %s", path)
        return None


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Replace one file only after its entire replacement has been flushed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_image(path: str | Path, image: np.ndarray) -> None:
    """Encode and atomically write an image, including to Unicode paths."""
    extension = Path(path).suffix.lower()
    if not extension:
        raise ValueError("An image output filename must have an extension")
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"Could not encode image for {path}")
    atomic_write_bytes(path, encoded.tobytes())


def copy_original_image(source: str | Path, destination: str | Path) -> bool:
    """Copy exact source bytes once; refuse to overwrite a different image.

    Return True if copied, False if destination is already this source or has
    identical bytes. No decoding, resizing, EXIF rewriting or re-encoding occurs.
    """
    source_path, destination_path = Path(source), Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.resolve() == destination_path.resolve():
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if filecmp.cmp(source_path, destination_path, shallow=False):
            return False
        raise FileExistsError(f"Refusing to overwrite a different original image: {destination_path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent)
    try:
        with source_path.open("rb") as source_stream, os.fdopen(fd, "wb") as output_stream:
            shutil.copyfileobj(source_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        shutil.copystat(source_path, temporary)
        # A hard link publishes the complete copy without replacing an existing
        # destination, including one created by another process during copying.
        if os.name == "nt":
            # Windows rename refuses an existing destination and also works on
            # removable/exFAT drives, which do not support hard links.
            os.rename(temporary, destination_path)
        else:
            os.link(temporary, destination_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True
