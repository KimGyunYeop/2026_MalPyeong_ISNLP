"""Hugging Face cache를 registry-safe Docker COPY layer들로 분할한다.

GHCR은 단일 layer가 10GB를 넘으면 push를 거부한다. 모델 shard 자체를 바꾸거나 다시
serialize하지 않고 cache file/symlink를 고정된 8개 source directory에 나눠 담는다. 각
snapshot symlink는 대상 blob과 같은 partition에 배치하므로 Docker build context 안에서도
끊어진 링크가 되지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PART_COUNT = 8
MAX_PARTITION_LAYER_BYTES = 8 * 2**30
GHCR_LAYER_LIMIT_BYTES = 10_000_000_000


class PartitionError(RuntimeError):
    """Cache partition 또는 layer audit 계약 위반."""


@dataclass
class CacheEntry:
    source: Path
    relative: Path
    kind: str
    size: int
    link_target: str | None = None

    @property
    def estimated_tar_bytes(self) -> int:
        # tar header/padding과 parent-directory metadata에 file당 4KiB를 보수적으로 잡는다.
        if self.kind == "symlink":
            return 4096
        return math.ceil(self.size / 512) * 512 + 4096


@dataclass
class EntryGroup:
    key: str
    entries: list[CacheEntry] = field(default_factory=list)

    @property
    def estimated_tar_bytes(self) -> int:
        return sum(entry.estimated_tar_bytes for entry in self.entries)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PartitionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise PartitionError(f"symlink가 cache root 밖을 가리킵니다: {path}") from exc


def _collect_entries(source_hub: Path) -> tuple[list[CacheEntry], list[EntryGroup]]:
    root = source_hub.resolve()
    entries: list[CacheEntry] = []
    regular_by_relative: dict[Path, CacheEntry] = {}
    symlinks: list[CacheEntry] = []

    for source in sorted(source_hub.rglob("*")):
        relative = source.relative_to(source_hub)
        _require(".." not in relative.parts, f"안전하지 않은 cache 경로: {relative}")
        if source.is_symlink():
            target = os.readlink(source)
            entry = CacheEntry(
                source=source,
                relative=relative,
                kind="symlink",
                size=len(os.fsencode(target)),
                link_target=target,
            )
            entries.append(entry)
            symlinks.append(entry)
        elif source.is_file():
            entry = CacheEntry(
                source=source,
                relative=relative,
                kind="file",
                size=source.stat().st_size,
            )
            entries.append(entry)
            regular_by_relative[relative] = entry
        elif source.is_dir():
            continue
        else:
            raise PartitionError(f"지원하지 않는 cache entry입니다: {source}")

    groups: dict[str, EntryGroup] = {
        relative.as_posix(): EntryGroup(relative.as_posix(), [entry])
        for relative, entry in regular_by_relative.items()
    }
    for entry in symlinks:
        assert entry.link_target is not None
        target_absolute = (entry.source.parent / entry.link_target).resolve()
        target_relative = _within(target_absolute, root)
        target_entry = regular_by_relative.get(target_relative)
        _require(
            target_entry is not None,
            f"cache symlink 대상 regular file이 없습니다: {entry.relative} -> "
            f"{entry.link_target}",
        )
        groups[target_relative.as_posix()].entries.append(entry)

    _require(
        sum(len(group.entries) for group in groups.values()) == len(entries),
        "cache entry grouping 중 중복/누락이 생겼습니다",
    )
    return entries, list(groups.values())


def _assign_groups(
    groups: Iterable[EntryGroup], *, part_count: int, max_layer_bytes: int
) -> list[list[EntryGroup]]:
    _require(part_count > 0, "part_count는 양수여야 합니다")
    _require(max_layer_bytes > 0, "max_layer_bytes는 양수여야 합니다")
    bins: list[list[EntryGroup]] = [[] for _ in range(part_count)]
    sizes = [0] * part_count
    for group in sorted(groups, key=lambda item: (-item.estimated_tar_bytes, item.key)):
        _require(
            group.estimated_tar_bytes < max_layer_bytes,
            "단일 blob+snapshot-link group이 partition 제한보다 큽니다: "
            f"{group.key}={group.estimated_tar_bytes} >= {max_layer_bytes}",
        )
        eligible = [
            index
            for index, size in enumerate(sizes)
            if size + group.estimated_tar_bytes < max_layer_bytes
        ]
        _require(
            bool(eligible),
            f"{part_count}개 partition에 cache를 {max_layer_bytes} byte 미만으로 담을 수 없습니다",
        )
        # 현재 가장 작은 bin에 넣어 shard가 고르게 분산되게 한다.
        chosen = min(eligible, key=lambda index: (sizes[index], index))
        bins[chosen].append(group)
        sizes[chosen] += group.estimated_tar_bytes
    return bins


def _remove_empty_source_tree(source_hub: Path) -> None:
    directories = [path for path in source_hub.rglob("*") if path.is_dir()]
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        directory.rmdir()
    source_hub.rmdir()


def partition_cache(
    source_hub: Path,
    parts_root: Path,
    manifest_path: Path,
    *,
    part_count: int = PART_COUNT,
    max_layer_bytes: int = MAX_PARTITION_LAYER_BYTES,
) -> dict[str, Any]:
    """Move one assembled cache into fixed partition roots and record hashes."""

    source_hub = source_hub.resolve()
    parts_root = parts_root.resolve()
    _require(source_hub.is_dir(), f"source cache가 없습니다: {source_hub}")
    _require(source_hub != parts_root, "source와 parts_root가 같습니다")
    _require(
        part_count == PART_COUNT,
        f"Dockerfile은 partition {PART_COUNT}개로 고정되어 있습니다: {part_count}",
    )
    parts_root.mkdir(parents=True, exist_ok=True)
    existing = list(parts_root.iterdir())
    _require(not existing, f"parts_root가 비어 있지 않습니다: {parts_root}")

    entries, groups = _collect_entries(source_hub)
    bins = _assign_groups(
        groups, part_count=part_count, max_layer_bytes=max_layer_bytes
    )
    partition_rows: list[dict[str, Any]] = []
    combined = hashlib.sha256()

    for index, groups_in_part in enumerate(bins):
        name = f"part{index:02d}"
        destination_root = parts_root / name
        destination_root.mkdir()
        records: list[dict[str, Any]] = []
        for group in sorted(groups_in_part, key=lambda item: item.key):
            # Blob을 먼저 옮겨 snapshot symlink가 destination context에서 즉시 유효하게 한다.
            ordered = sorted(
                group.entries,
                key=lambda entry: (entry.kind == "symlink", entry.relative.as_posix()),
            )
            for entry in ordered:
                destination = destination_root / entry.relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                _require(
                    not os.path.lexists(destination),
                    f"partition 경로 충돌: {destination}",
                )
                entry.source.rename(destination)
                if entry.kind == "file":
                    digest = _sha256(destination)
                    record = {
                        "path": entry.relative.as_posix(),
                        "kind": "file",
                        "bytes": entry.size,
                        "sha256": digest,
                    }
                else:
                    _require(
                        destination.is_symlink(), f"symlink 이동 실패: {destination}"
                    )
                    actual_target = os.readlink(destination)
                    _require(
                        actual_target == entry.link_target,
                        f"symlink target 변경: {destination}: {actual_target!r}",
                    )
                    record = {
                        "path": entry.relative.as_posix(),
                        "kind": "symlink",
                        "bytes": entry.size,
                        "target": actual_target,
                    }
                records.append(record)
                combined.update(name.encode("ascii"))
                combined.update(b"\0")
                combined.update(json.dumps(record, sort_keys=True).encode("utf-8"))
                combined.update(b"\n")
        estimated = sum(group.estimated_tar_bytes for group in groups_in_part)
        file_bytes = sum(
            entry.size
            for group in groups_in_part
            for entry in group.entries
            if entry.kind == "file"
        )
        _require(
            estimated < max_layer_bytes,
            f"{name} layer 추정치가 제한을 넘습니다: {estimated} >= {max_layer_bytes}",
        )
        partition_rows.append(
            {
                "name": name,
                "file_bytes": file_bytes,
                "estimated_layer_bytes": estimated,
                "entry_count": len(records),
                "entries": records,
            }
        )

    _remove_empty_source_tree(source_hub)
    manifest = {
        "schema_version": 1,
        "algorithm": "blob_with_snapshot_symlinks_greedy_v1",
        "part_count": part_count,
        "max_partition_layer_bytes": max_layer_bytes,
        "ghcr_layer_limit_bytes": GHCR_LAYER_LIMIT_BYTES,
        "total_file_bytes": sum(
            entry.size for entry in entries if entry.kind == "file"
        ),
        "total_entry_count": len(entries),
        "combined_sha256": combined.hexdigest(),
        "parts": partition_rows,
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartitionError(f"partition manifest를 읽을 수 없습니다: {path}") from exc
    _require(isinstance(value, dict), "partition manifest root는 object여야 합니다")
    return value


def verify_partitions(parts_root: Path, manifest: dict[str, Any]) -> None:
    """Manifest와 source partition의 경로, size, hash, symlink를 전부 대조한다."""

    _require(
        manifest.get("part_count") == PART_COUNT,
        f"Dockerfile 고정 partition 수와 manifest가 다릅니다: {manifest.get('part_count')}",
    )
    expected_parts = [f"part{index:02d}" for index in range(PART_COUNT)]
    actual_parts = sorted(path.name for path in parts_root.iterdir() if path.is_dir())
    _require(
        actual_parts == expected_parts,
        f"partition directory가 다릅니다: {actual_parts}",
    )
    expected_paths: set[tuple[str, str]] = set()
    manifest_parts = manifest.get("parts")
    _require(isinstance(manifest_parts, list), "manifest.parts는 list여야 합니다")
    manifest_part_names = [part.get("name") for part in manifest_parts]
    _require(
        manifest_part_names == expected_parts,
        f"manifest partition 순서/집합이 다릅니다: {manifest_part_names}",
    )

    for part in manifest_parts:
        name = part.get("name")
        _require(name in expected_parts, f"잘못된 partition 이름: {name!r}")
        estimated = int(part.get("estimated_layer_bytes", -1))
        limit = int(manifest.get("max_partition_layer_bytes", -1))
        _require(0 <= estimated < limit, f"{name} manifest layer 제한 위반")
        root = parts_root / name
        for record in part.get("entries", []):
            relative = Path(str(record.get("path", "")))
            _require(
                relative.parts
                and not relative.is_absolute()
                and ".." not in relative.parts,
                f"안전하지 않은 partition entry: {relative}",
            )
            key = (name, relative.as_posix())
            _require(key not in expected_paths, f"중복 partition entry: {key}")
            expected_paths.add(key)
            path = root / relative
            if record.get("kind") == "file":
                _require(
                    path.is_file() and not path.is_symlink(),
                    f"regular file 없음: {path}",
                )
                _require(
                    path.stat().st_size == record.get("bytes"),
                    f"file size 불일치: {path}",
                )
                _require(
                    _sha256(path) == record.get("sha256"), f"file hash 불일치: {path}"
                )
            elif record.get("kind") == "symlink":
                _require(path.is_symlink(), f"symlink 없음: {path}")
                _require(
                    os.readlink(path) == record.get("target"),
                    f"symlink target 불일치: {path}",
                )
                target = (path.parent / os.readlink(path)).resolve()
                _require(
                    target.is_file(),
                    f"partition 안에서 symlink가 끊어졌습니다: {path} -> {os.readlink(path)}",
                )
            else:
                raise PartitionError(f"알 수 없는 entry kind: {record.get('kind')!r}")

    actual_paths: set[tuple[str, str]] = set()
    for name in expected_parts:
        root = parts_root / name
        for path in root.rglob("*"):
            if path.is_symlink() or path.is_file():
                actual_paths.add((name, path.relative_to(root).as_posix()))
    _require(
        actual_paths == expected_paths,
        f"partition manifest와 실제 entry 집합이 다릅니다: "
        f"missing={sorted(expected_paths - actual_paths)}, "
        f"extra={sorted(actual_paths - expected_paths)}",
    )


def assemble_partitions(
    parts_root: Path, destination_hub: Path, manifest: dict[str, Any]
) -> None:
    """Docker의 반복 COPY 결과를 hardlink 기반 임시 tree로 재현한다."""

    verify_partitions(parts_root, manifest)
    _require(
        not destination_hub.exists(),
        f"assembly destination이 이미 있습니다: {destination_hub}",
    )
    destination_hub.mkdir(parents=True)
    for part in manifest["parts"]:
        root = parts_root / part["name"]
        for record in part["entries"]:
            relative = Path(record["path"])
            source = root / relative
            destination = destination_hub / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _require(
                not os.path.lexists(destination), f"assembly 경로 충돌: {destination}"
            )
            if record["kind"] == "file":
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            else:
                destination.symlink_to(record["target"])
    for path in destination_hub.rglob("*"):
        if path.is_symlink():
            _require(
                path.resolve().is_file(),
                f"assembled cache symlink가 끊어졌습니다: {path}",
            )


_SIZE_PATTERN = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(B|kB|MB|GB|TB|KiB|MiB|GiB|TiB)$")


def parse_docker_size(value: str) -> int:
    match = _SIZE_PATTERN.fullmatch(value.strip())
    if not match:
        raise PartitionError(
            f"Docker history size 형식을 해석할 수 없습니다: {value!r}"
        )
    amount = float(match.group(1))
    unit = match.group(2)
    multipliers = {
        "B": 1,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KiB": 2**10,
        "MiB": 2**20,
        "GiB": 2**30,
        "TiB": 2**40,
    }
    return int(amount * multipliers[unit])


def audit_docker_history(
    history_path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Build 결과의 모든 layer와 고정 HF COPY layer 수/크기를 fail-close한다."""

    rows = []
    with history_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PartitionError(
                    f"Docker history JSONL {line_number} 파싱 실패"
                ) from exc
            _require(isinstance(row, dict), "Docker history row가 object가 아닙니다")
            size = parse_docker_size(str(row.get("Size", "")))
            created_by = str(row.get("CreatedBy", ""))
            _require(
                size < GHCR_LAYER_LIMIT_BYTES,
                f"GHCR 10GB 제한을 넘는 최종 image layer: {size} bytes: {created_by}",
            )
            rows.append({"size_bytes": size, "created_by": created_by})

    expected_names = [f"part{index:02d}" for index in range(PART_COUNT)]
    seen: dict[str, list[int]] = {name: [] for name in expected_names}
    for row in rows:
        for name in expected_names:
            marker = f"COPY submission_assets/hf_parts/{name}/"
            if marker in row["created_by"]:
                seen[name].append(row["size_bytes"])
    for part in manifest["parts"]:
        name = part["name"]
        _require(
            len(seen[name]) == 1,
            f"Docker history에 {name} COPY가 정확히 하나여야 합니다",
        )
        _require(
            seen[name][0] < int(manifest["max_partition_layer_bytes"]),
            f"최종 {name} layer가 8GiB partition 제한을 넘습니다: {seen[name][0]}",
        )
    return {
        "schema_version": 1,
        "layer_count": len(rows),
        "maximum_layer_bytes": max((row["size_bytes"] for row in rows), default=0),
        "ghcr_layer_limit_bytes": GHCR_LAYER_LIMIT_BYTES,
        "hf_copy_layers": {name: values[0] for name, values in seen.items()},
        "all_layers_below_ghcr_limit": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--parts-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    audit = subparsers.add_parser("audit-history")
    audit.add_argument("--history", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    if args.command == "verify":
        verify_partitions(args.parts_root, manifest)
        print(f"PASS: {args.parts_root}")
        return 0
    report = audit_docker_history(args.history, manifest)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: {report['layer_count']} layers, max={report['maximum_layer_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
