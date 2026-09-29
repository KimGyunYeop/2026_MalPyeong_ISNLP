from __future__ import annotations

import json
from pathlib import Path

import pytest

from main_code_relonation.prompts import (
    baseline_prompt_template,
    prompt_template_sha256,
)
from main_code_submission.rationale_prompt_binding import (
    bind_adapter_prompt_to_manifest,
    prompt_binding_from_adapter,
)


def _legacy_manifest(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payload = {
        "name": "legacy-prompt",
        "root": str(tmp_path),
        "score_postprocess": "average_matched",
        "score_members": [
            {
                "name": "m0",
                "checkpoint": "checkpoint",
                "backbone_key": "m0",
                "parameters_billion": 1.0,
            }
        ],
        "rationale": {"base_model": "stub", "enabled": False},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_legacy_manifest_without_prompt_fields_is_exact_baseline(
    tmp_path: Path,
) -> None:
    from main_code_submission.config import load_manifest

    config = load_manifest(_legacy_manifest(tmp_path))
    expected = baseline_prompt_template()
    assert config.rationale.rationale_prompt_id == "baseline_prompt"
    assert config.rationale.rationale_prompt_text == expected
    assert config.rationale.rationale_prompt_sha256 == prompt_template_sha256(expected)


def test_manifest_prompt_text_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    from main_code_submission.config import load_manifest

    path = _legacy_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rationale"].update(
        {
            "rationale_prompt_id": "v1",
            "rationale_prompt_text": baseline_prompt_template(),
            "rationale_prompt_sha256": "0" * 64,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt 해시가 원문과 다릅니다"):
        load_manifest(path)


def test_old_adapter_without_sidecars_uses_baseline_without_mutation(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    marker = adapter / "adapter_model.safetensors"
    marker.write_bytes(b"old")
    before = {path.name: path.read_bytes() for path in adapter.iterdir()}

    binding = prompt_binding_from_adapter(adapter)

    assert binding.prompt_id == "baseline_prompt"
    assert binding.text == baseline_prompt_template()
    assert {path.name: path.read_bytes() for path in adapter.iterdir()} == before


def test_new_adapter_sidecars_are_bound_verbatim_to_staged_manifest(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    text = baseline_prompt_template() + "\n"
    sha256 = prompt_template_sha256(text)
    (adapter / "rationale_prompt.txt").write_text(text, encoding="utf-8")
    (adapter / "rationale_runtime_config.json").write_text(
        json.dumps(
            {
                "rationale_prompt_id": "test-v1",
                "rationale_prompt_text": text,
                "rationale_prompt_sha256": sha256,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    staged: dict[str, object] = {}

    binding = bind_adapter_prompt_to_manifest(staged, adapter)

    assert binding.prompt_id == "test-v1"
    assert staged == {
        "rationale_prompt_id": "test-v1",
        "rationale_prompt_text": text,
        "rationale_prompt_sha256": sha256,
    }


def test_adapter_prompt_file_and_runtime_config_must_match(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    text = baseline_prompt_template()
    (adapter / "rationale_prompt.txt").write_text(text, encoding="utf-8")
    (adapter / "rationale_runtime_config.json").write_text(
        json.dumps(
            {
                "rationale_prompt_id": "v1",
                "rationale_prompt_text": text + "changed",
                "rationale_prompt_sha256": prompt_template_sha256(text + "changed"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prompt text.*다릅니다"):
        prompt_binding_from_adapter(adapter)


def test_docker_stager_binds_prompt_before_copying_adapter() -> None:
    source = (Path(__file__).parents[1] / "build_image.sh").read_text(encoding="utf-8")
    assert "bind_adapter_prompt_to_manifest(rationale, source)" in source
    assert source.index(
        "bind_adapter_prompt_to_manifest(rationale, source)"
    ) < source.index("shutil.copytree(\n        source,\n        target,")
