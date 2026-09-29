"""제출 checkpoint/adapter의 결정적 SHA-256 지문.

Host manifest의 원본 provenance와 Docker staging 뒤 실제 배포 artifact를 같은 규칙으로
구분해 기록한다. 경로는 artifact root에 상대적인 POSIX 경로만 hash에 넣고, 파일 내용과
크기를 함께 묶어 파일 추가/삭제/교체를 모두 검출한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Collection


class ArtifactIntegrityError(ValueError):
    """Artifact가 지문 계약을 만족하지 않을 때 발생한다."""


_CHECKPOINT_ARTIFACT_CANDIDATES = (
    "config.json",
    "heads.pt",
    "manifest.json",
    "selection.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "adapter/adapter_model.bin",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_files(root: Path, files: list[Path]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        hexdigest = _file_sha256(path)
        size = path.stat().st_size
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(hexdigest.encode("ascii"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\n")
        entries.append({"path": relative, "sha256": hexdigest, "bytes": size})
    return {"combined_sha256": combined.hexdigest(), "files": entries}


def checkpoint_fingerprint(checkpoint: str | Path) -> dict[str, Any]:
    """점수 배포 의미를 정하는 기존 핵심 파일 지문을 계산한다.

    기존 후보 manifest와의 provenance 호환을 위해 이 목록은 의도적으로 고정한다. 전체
    파일 closure는 :func:`artifact_tree_fingerprint`로 별도 기록·검증한다.
    """

    root = Path(checkpoint)
    if not root.is_dir():
        raise ArtifactIntegrityError(f"checkpoint 폴더가 없습니다: {root}")
    files = [root / relative for relative in _CHECKPOINT_ARTIFACT_CANDIDATES]
    files = [path for path in files if path.is_file() and not path.is_symlink()]
    if not any(path.name.startswith("adapter_model") for path in files):
        raise ArtifactIntegrityError(f"checkpoint adapter weight가 없습니다: {root}")
    return _fingerprint_files(root, files)


def artifact_tree_fingerprint(
    artifact_root: str | Path,
    *,
    excluded_relative_paths: Collection[str] = (),
) -> dict[str, Any]:
    """Artifact tree의 모든 일반 파일을 지문화하고 안전하지 않은 entry를 거부한다."""

    root = Path(artifact_root)
    if not root.is_dir():
        raise ArtifactIntegrityError(f"artifact 폴더가 없습니다: {root}")
    excluded = {Path(item).as_posix() for item in excluded_relative_paths}
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ArtifactIntegrityError(
                f"artifact 안의 symlink는 허용하지 않습니다: {relative}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactIntegrityError(
                f"artifact 안의 특수 파일은 허용하지 않습니다: {relative}"
            )
        files.append(path)
    if not files:
        raise ArtifactIntegrityError(f"artifact 파일이 없습니다: {root}")
    return _fingerprint_files(root, files)


def require_fingerprint(
    *,
    label: str,
    expected: object,
    actual: dict[str, Any],
) -> None:
    """Manifest 지문과 실제 artifact가 문자 단위로 같지 않으면 fail-close한다."""

    if expected != actual:
        expected_sha = (
            expected.get("combined_sha256") if isinstance(expected, dict) else None
        )
        raise ArtifactIntegrityError(
            f"{label} 지문 불일치: expected={expected_sha!r}, "
            f"actual={actual['combined_sha256']!r}"
        )
