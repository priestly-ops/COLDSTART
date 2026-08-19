"""Helpers for experiment reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "h5py",
    "joblib",
    "fastdtw",
    "pytest",
    "pyyaml",
)


def file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: str | Path | None = None) -> str | None:
    return _run_git(("rev-parse", "HEAD"), repo_root)


def git_dirty(repo_root: str | Path | None = None) -> bool | None:
    status = _run_git(("status", "--porcelain"), repo_root)
    if status is None:
        return None
    return bool(status.strip())


def package_versions(
    package_names: Iterable[str] = DEFAULT_PACKAGE_NAMES,
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def reproducibility_metadata(
    *,
    repo_root: str | Path | None = None,
    input_paths: dict[str, str | Path] | None = None,
    artifact_paths: dict[str, str | Path] | None = None,
    package_names: Iterable[str] = DEFAULT_PACKAGE_NAMES,
) -> dict[str, object]:
    inputs = {
        key: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for key, path in (input_paths or {}).items()
    }
    artifacts = {
        key: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for key, path in (artifact_paths or {}).items()
    }
    return {
        "git_commit": git_commit(repo_root),
        "git_dirty": git_dirty(repo_root),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": package_versions(package_names),
        "input_hashes": inputs,
        "artifact_hashes": artifacts,
    }


def _run_git(args: tuple[str, ...], repo_root: str | Path | None) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=None if repo_root is None else Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
