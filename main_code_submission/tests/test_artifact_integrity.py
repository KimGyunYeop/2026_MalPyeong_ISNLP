from __future__ import annotations

import json
from pathlib import Path

import pytest

from main_code_submission.artifact_integrity import (
    ArtifactIntegrityError,
    artifact_tree_fingerprint,
    checkpoint_fingerprint,
)
from main_code_submission.config import load_manifest


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _checkpoint(root: Path) -> None:
    _write(root / "config.json", b'{"model_id":"stub"}\n')
    _write(root / "manifest.json", b'{"checkpoint":true}\n')
    _write(root / "selection.json", b'{"metric":"rmse"}\n')
    _write(root / "heads.pt", b"heads")
    _write(root / "adapter" / "adapter_config.json", b'{"r":32}\n')
    _write(root / "adapter" / "adapter_model.safetensors", b"adapter")
    _write(root / "tokenizer" / "tokenizer_config.json", b"{}\n")


def _rationale(root: Path) -> None:
    _write(root / "adapter_config.json", b'{"task_type":"CAUSAL_LM"}\n')
    _write(root / "adapter_model.safetensors", b"rationale")
    _write(root / "tokenizer_config.json", b"{}\n")


def _staged_manifest(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoints" / "m0"
    rationale = tmp_path / "rationale_adapter"
    _checkpoint(checkpoint)
    _rationale(rationale)
    payload = {
        "name": "integrity",
        "root": str(tmp_path),
        "served_model_name": "stub",
        "score_postprocess": "average_matched",
        "essay_surface": "official_raw",
        "score_members": [
            {
                "name": "m0",
                "checkpoint": "checkpoints/m0",
                "backbone_key": "m0",
                "parameters_billion": 1.0,
            }
        ],
        "rationale": {
            "base_model": "stub",
            "adapter": "rationale_adapter",
            "share_backbone_key": "m0",
            "enabled": True,
        },
        "extra": {
            "artifact_integrity_required": True,
            "deployed_checkpoint_artifacts": {"m0": checkpoint_fingerprint(checkpoint)},
            "deployed_checkpoint_closures": {
                "m0": artifact_tree_fingerprint(checkpoint)
            },
            "deployed_rationale_artifact": artifact_tree_fingerprint(rationale),
        },
    }
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_core_fingerprint_and_full_closure_have_distinct_jobs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint)
    original_core = checkpoint_fingerprint(checkpoint)
    original_closure = artifact_tree_fingerprint(checkpoint)

    # Tokenizer도 런타임 asset이므로 전체 closure는 잡되, 기존 provenance core 목록은
    # 의도적으로 그대로다.
    _write(checkpoint / "tokenizer" / "added_tokens.json", b"{}\n")
    assert checkpoint_fingerprint(checkpoint) == original_core
    assert artifact_tree_fingerprint(checkpoint) != original_closure

    _write(checkpoint / "config.json", b'{"model_id":"scrubbed"}\n')
    assert checkpoint_fingerprint(checkpoint) != original_core


def test_staged_manifest_self_attestation_passes_then_fails_closed(
    tmp_path: Path,
) -> None:
    manifest = _staged_manifest(tmp_path)
    assert load_manifest(manifest).name == "integrity"

    # Manifest 밖 checkpoint tree의 어느 runtime 파일도 조용히 추가될 수 없다.
    _write(tmp_path / "checkpoints" / "m0" / "unexpected.bin", b"unexpected")
    with pytest.raises(ArtifactIntegrityError, match="closure.*지문 불일치"):
        load_manifest(manifest)


def test_container_manifest_cannot_disable_integrity_gate(tmp_path: Path) -> None:
    manifest = tmp_path / "submission_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "missing-integrity",
                "root": "/opt/submission",
                "score_postprocess": "average_matched",
                "score_members": [],
                "rationale": {"enabled": False},
                "extra": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact_integrity_required=true"):
        load_manifest(manifest)


def test_deployed_manifest_cannot_disable_rationale(tmp_path: Path) -> None:
    manifest = _staged_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rationale"]["enabled"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="rationale adapter를 활성화"):
        load_manifest(manifest)


def test_staged_manifest_detects_rationale_mutation(tmp_path: Path) -> None:
    manifest = _staged_manifest(tmp_path)
    _write(tmp_path / "rationale_adapter" / "adapter_model.safetensors", b"changed")
    with pytest.raises(ArtifactIntegrityError, match="rationale adapter 지문 불일치"):
        load_manifest(manifest)


def test_build_staging_scrubs_paths_and_training_metadata_after_copy() -> None:
    root = Path(__file__).parents[2]
    source = (root / "main_code_submission" / "build_image.sh").read_text(
        encoding="utf-8"
    )
    assert 'extra.pop("source_run", None)' in source
    assert 'ignore=shutil.ignore_patterns("training_args.bin")' in source
    assert '"artifact_integrity_required": True' in source
    assert source.index("stage_checkpoint(source, target)") < source.index(
        'deployed_checkpoint_artifacts[member["name"]] = checkpoint_fingerprint'
    )


def test_runtime_integrity_helper_is_copied_into_image() -> None:
    root = Path(__file__).parents[2]
    dockerfile = (root / "main_code_submission" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    assert "main_code_submission/artifact_integrity.py" in dockerfile
    assert "!main_code_submission/artifact_integrity.py" in dockerignore
    assert "submission_assets/rationale_adapter/training_args.bin" in dockerignore
    assert "submission_assets/rationale_adapter/**/training_args.bin" in dockerignore
