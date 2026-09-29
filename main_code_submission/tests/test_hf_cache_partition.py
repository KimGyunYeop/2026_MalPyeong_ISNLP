from __future__ import annotations

import json
from pathlib import Path

import pytest

from main_code_submission.hf_cache_partition import (
    GHCR_LAYER_LIMIT_BYTES,
    MAX_PARTITION_LAYER_BYTES,
    PART_COUNT,
    PartitionError,
    assemble_partitions,
    audit_docker_history,
    partition_cache,
    verify_partitions,
)


def _fake_cache(root: Path) -> dict[str, bytes]:
    repo = root / "models--owner--model"
    blobs = {
        "a" * 64: b"a" * 100,
        "b" * 64: b"b" * 120,
        "config": b'{"model_type":"test"}',
    }
    for name, payload in blobs.items():
        path = repo / "blobs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    snapshot = repo / "snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model-00001.safetensors").symlink_to(f"../../blobs/{'a' * 64}")
    (snapshot / "model-00002.safetensors").symlink_to(f"../../blobs/{'b' * 64}")
    (snapshot / "config.json").symlink_to("../../blobs/config")
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text("c" * 40, encoding="utf-8")
    return blobs


def test_partition_reassembly_preserves_final_cache_semantics(tmp_path: Path) -> None:
    source = tmp_path / "hf" / "hub"
    expected_blobs = _fake_cache(source)
    parts = tmp_path / "hf_parts"
    manifest_path = tmp_path / "hf_partition_manifest.json"

    manifest = partition_cache(
        source,
        parts,
        manifest_path,
        # blob+그 snapshot symlink가 한 part에 같이 들어가는지를 작은 limit로 확인한다.
        max_layer_bytes=10_000,
    )

    assert not source.exists()
    assert manifest["part_count"] == PART_COUNT
    assert [part["name"] for part in manifest["parts"]] == [
        f"part{index:02d}" for index in range(PART_COUNT)
    ]
    assert all(part["estimated_layer_bytes"] < 10_000 for part in manifest["parts"])
    verify_partitions(parts, manifest)

    # 각 snapshot link와 그 blob은 Docker COPY source 하나 안에서도 유효해야 한다.
    for part in manifest["parts"]:
        part_root = parts / part["name"]
        for record in part["entries"]:
            path = part_root / record["path"]
            if record["kind"] == "symlink":
                assert path.resolve().is_file()

    assembled = tmp_path / "assembled" / "hub"
    assemble_partitions(parts, assembled, manifest)
    repo = assembled / "models--owner--model"
    assert (repo / "refs" / "main").read_text(encoding="utf-8") == "c" * 40
    assert (
        repo / "snapshots" / ("c" * 40) / "model-00001.safetensors"
    ).read_bytes() == expected_blobs["a" * 64]
    assert (
        repo / "snapshots" / ("c" * 40) / "model-00002.safetensors"
    ).read_bytes() == expected_blobs["b" * 64]
    assert (
        repo / "snapshots" / ("c" * 40) / "config.json"
    ).read_bytes() == expected_blobs["config"]


def test_runtime_empty_cache_still_has_all_fixed_copy_sources(tmp_path: Path) -> None:
    source = tmp_path / "hf" / "hub"
    source.mkdir(parents=True)
    parts = tmp_path / "hf_parts"
    manifest = partition_cache(source, parts, tmp_path / "manifest.json")

    assert manifest["total_entry_count"] == 0
    assert sorted(path.name for path in parts.iterdir()) == [
        f"part{index:02d}" for index in range(PART_COUNT)
    ]
    verify_partitions(parts, manifest)
    assembled = tmp_path / "assembled" / "hub"
    assemble_partitions(parts, assembled, manifest)
    assert list(assembled.iterdir()) == []


def test_single_blob_group_over_limit_fails_before_moving_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hub"
    _fake_cache(source)
    with pytest.raises(PartitionError, match="단일 blob"):
        partition_cache(
            source,
            tmp_path / "parts",
            tmp_path / "manifest.json",
            max_layer_bytes=8_000,
        )
    assert source.is_dir()
    assert any(source.rglob("*"))


def _history_rows(*, missing: str | None = None, oversized: str | None = None) -> str:
    rows = [
        {
            "Size": "6.11GB",
            "CreatedBy": "COPY /opt/conda /opt/conda # buildkit",
        }
    ]
    for index in range(PART_COUNT):
        name = f"part{index:02d}"
        if name == missing:
            continue
        rows.append(
            {
                "Size": "8.60GB" if name == oversized else "4.70GB",
                "CreatedBy": (
                    f"COPY submission_assets/hf_parts/{name}/ "
                    "/opt/submission/hf/hub/ # buildkit"
                ),
            }
        )
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def _history_manifest() -> dict[str, object]:
    return {
        "part_count": PART_COUNT,
        "max_partition_layer_bytes": MAX_PARTITION_LAYER_BYTES,
        "parts": [{"name": f"part{index:02d}"} for index in range(PART_COUNT)],
    }


def test_final_docker_history_requires_eight_small_hf_layers(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(_history_rows(), encoding="utf-8")
    report = audit_docker_history(history, _history_manifest())
    assert report["all_layers_below_ghcr_limit"] is True
    assert report["maximum_layer_bytes"] < GHCR_LAYER_LIMIT_BYTES
    assert len(report["hf_copy_layers"]) == PART_COUNT

    history.write_text(_history_rows(missing="part07"), encoding="utf-8")
    with pytest.raises(PartitionError, match="part07 COPY"):
        audit_docker_history(history, _history_manifest())

    # 8.60 decimal GB는 8GiB보다 크지만 GHCR 10GB보다는 작다. HF 자체 buffer gate가 막는다.
    history.write_text(_history_rows(oversized="part03"), encoding="utf-8")
    with pytest.raises(PartitionError, match="8GiB"):
        audit_docker_history(history, _history_manifest())


def test_dockerfile_orders_common_large_layers_before_candidates() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "main_code_submission" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    positions = [
        dockerfile.index(f"COPY submission_assets/hf_parts/part{index:02d}/")
        for index in range(PART_COUNT)
    ]
    assert positions == sorted(positions)
    rationale = dockerfile.index("COPY submission_assets/rationale_adapter/")
    checkpoint = dockerfile.index("COPY submission_assets/checkpoints/")
    manifest = dockerfile.index("COPY submission_assets/submission_manifest.json")
    assert positions[-1] < rationale < checkpoint < manifest
    assert "COPY submission_assets/hf/hub/" not in dockerfile

    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    for index in range(PART_COUNT):
        assert f"!submission_assets/hf_parts/part{index:02d}/**" in dockerignore
    assert "!submission_assets/hf/hub/**" not in dockerignore
