"""Assemble a self-contained, traceable Windows release on GitHub Actions."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"(?:v1\.9\.0[A-Za-z0-9._-]*|build-[0-9]+)", args.tag):
        parser.error("Invalid release tag for a portable archive filename")
    root = Path(__file__).resolve().parents[1]
    bundle = root / "dist" / "VisionAce-Improved"
    output = root / "release"
    report_dir = root / "reports"
    output.mkdir(exist_ok=True)
    if not (bundle / "VisionAce-Improved.exe").is_file():
        raise RuntimeError("The frozen executable is missing")
    smoke = json.loads((report_dir / "frozen-smoke.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "passed" or smoke.get("frozen") is not True:
        raise RuntimeError("A passing frozen-executable smoke report is required")
    if smoke.get("cuda_runtime") is not None:
        raise RuntimeError("This release must use CPU-only PyTorch wheels")

    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    (bundle / "requirements-resolved.txt").write_text(freeze, encoding="utf-8")
    provenance = {
        "release": args.tag, "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "commit": os.environ.get("GITHUB_SHA", ""),
        "workflow_run": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}",
        "runner": os.environ.get("RUNNER_OS", platform.system()),
        "python": sys.version, "platform": platform.platform(),
        "build_flavor": "Windows x64 CPU portable onedir",
        "model_weights_included": False, "gpu_runtime_included": False,
        "tensorflow_included": False,
        "validation": "GitHub-hosted unit tests and frozen startup/runtime smoke only; no user model accuracy validation",
    }
    (bundle / "build-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for source in [root / "README.md", root / "CHANGELOG.md", root / "THIRD_PARTY_NOTICES.md", root / "LICENSE"]:
        if source.is_file():
            shutil.copy2(source, bundle / source.name)
    if (root / "docs").is_dir():
        shutil.copytree(root / "docs", bundle / "docs", dirs_exist_ok=True)
    (bundle / "START_HERE.txt").write_text(
        "VisionAce Improved - Windows 10/11 x64 CPU portable\n\n"
        "1. Extract the entire ZIP to a writable folder.\n"
        "2. Open VisionAce-Improved.exe; keep _internal next to it.\n"
        "Python and pip are not required on the destination PC.\n"
        "Bring your own trusted Ultralytics YOLO/RT-DETR .pt checkpoint.\n"
        "No model weights are included. No GPU/CUDA or TensorFlow runtime is included.\n"
        "DINOv3 backbone-only checkpoints do not define an object detector.\n"
        "Logs: %USERPROFILE%\\.visionace-improved\\logs\\visionace.log\n"
        "See README.md and build-provenance.json for validation limits.\n",
        encoding="utf-8",
    )

    # Retain installed distributions' notices/licenses alongside their version inventory.
    licenses = bundle / "third-party-licenses"
    licenses.mkdir(exist_ok=True)
    inventory = []
    for dist in sorted(metadata.distributions(), key=lambda item: item.metadata.get("Name", "").lower()):
        name = dist.metadata.get("Name", "unknown")
        entry = {"name": name, "version": dist.version, "license": dist.metadata.get("License-Expression") or dist.metadata.get("License", ""), "copied_files": []}
        for file in dist.files or []:
            filename = Path(str(file)).name
            if filename.lower().startswith(("license", "licence", "copying", "notice", "copyright")):
                src = Path(dist.locate_file(file))
                if src.is_file():
                    # Flatten package-relative paths safely into a distribution subfolder.
                    relative = str(file).replace("\\", "_").replace("/", "_").replace(":", "_")
                    dest = licenses / name / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    entry["copied_files"].append(str(dest.relative_to(bundle)))
        inventory.append(entry)
    (licenses / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    verification = bundle / "verification"
    shutil.copytree(report_dir, verification, dirs_exist_ok=True)

    manifest = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "FILES.sha256":
            manifest.append(f"{sha256(path)}  {path.relative_to(bundle).as_posix()}")
    (bundle / "FILES.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    archive = output / f"VisionAce-Improved-{args.tag}-Windows-x64-CPU.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zipped:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(bundle.parent).as_posix())
    if archive.stat().st_size >= 2_000_000_000:
        raise RuntimeError("Portable archive exceeds GitHub's per-asset limit")
    (output / "SHA256SUMS.txt").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    shutil.copy2(bundle / "build-provenance.json", output / "build-provenance.json")
    shutil.copy2(bundle / "requirements-resolved.txt", output / "requirements-resolved.txt")
    print(f"Packaged {archive.name}: {archive.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
