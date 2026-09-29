from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from main_code.config import RegressionConfig
from main_code_submission.config import (
    BF16_GIB_PER_BILLION,
    NF4_GIB_PER_BILLION,
    RationaleSpec,
    ScoreMember,
    SubmissionConfig,
    load_manifest,
)
from main_code_submission.engine import SubmissionEngine


def _checkpoint(root: Path, *, use_qlora: bool | None) -> Path:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    payload = {} if use_qlora is None else {"use_qlora": use_qlora}
    (checkpoint / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return checkpoint


def _manifest(
    root: Path,
    *,
    parameters_billion: float,
    declared_load_in_4bit: bool | None = None,
) -> Path:
    member: dict[str, object] = {
        "name": "score",
        "checkpoint": "checkpoint",
        "backbone_key": "score-backbone",
        "parameters_billion": parameters_billion,
        "weight": 1.0,
    }
    if declared_load_in_4bit is not None:
        member["load_in_4bit"] = declared_load_in_4bit
    payload = {
        "name": "qlora-contract",
        "root": str(root),
        "score_postprocess": "average_matched",
        "score_members": [member],
        "rationale": {"base_model": "stub", "enabled": False},
    }
    path = root / "submission.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_infers_nf4_from_checkpoint_and_budget_uses_it(
    tmp_path: Path,
) -> None:
    _checkpoint(tmp_path, use_qlora=True)
    config = load_manifest(_manifest(tmp_path, parameters_billion=30.0))

    assert config.score_members[0].load_in_4bit is True
    assert config.estimated_weights_gib() == pytest.approx(30.0 * NF4_GIB_PER_BILLION)
    report = config.vram_budget_report()
    assert report["score_load_in_4bit"] == {"score": True}
    assert report["fits"] is True


def test_historical_non_qlora_manifest_remains_bf16(tmp_path: Path) -> None:
    # Historical configs without the additive field mean the old BF16 path.
    _checkpoint(tmp_path, use_qlora=None)
    config = load_manifest(_manifest(tmp_path, parameters_billion=7.0))

    assert config.score_members[0].load_in_4bit is False
    assert config.estimated_weights_gib() == pytest.approx(7.0 * BF16_GIB_PER_BILLION)


def test_manifest_precision_mismatch_fails_before_model_load(tmp_path: Path) -> None:
    _checkpoint(tmp_path, use_qlora=True)
    path = _manifest(
        tmp_path,
        parameters_billion=9.0,
        declared_load_in_4bit=False,
    )
    with pytest.raises(ValueError, match="manifest load_in_4bit=False.*use_qlora=True"):
        load_manifest(path)


def test_bf16_score_over_static_l40s_budget_fails_closed(tmp_path: Path) -> None:
    _checkpoint(tmp_path, use_qlora=False)
    with pytest.raises(ValueError, match="L40S static VRAM budget"):
        load_manifest(_manifest(tmp_path, parameters_billion=30.0))


def test_single_member_engine_passes_manifest_nf4_to_checkpoint_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path, use_qlora=True)
    member = ScoreMember(
        name="score",
        checkpoint=checkpoint,
        backbone_key="score-backbone",
        parameters_billion=9.0,
        load_in_4bit=True,
    )
    submission = SubmissionConfig(
        name="engine-qlora",
        score_members=(member,),
        rationale=RationaleSpec(base_model="stub", adapter=None, enabled=False),
    )
    engine = SubmissionEngine(submission, device="cpu")
    checkpoint_config = RegressionConfig().with_updates(use_qlora=True)
    tokenizer = SimpleNamespace(model_max_length=512)
    calls: list[dict[str, object]] = []

    def fake_load_checkpoint(path: Path, **kwargs: object) -> SimpleNamespace:
        assert path == checkpoint
        calls.append(kwargs)
        return SimpleNamespace(
            scorer=object(), tokenizer=tokenizer, config=checkpoint_config
        )

    monkeypatch.setattr(
        "main_code_submission.engine.load_checkpoint", fake_load_checkpoint
    )
    loaded = engine._load_member(member)

    assert loaded.config.use_qlora is True
    assert calls == [{"device": engine.device, "load_in_4bit": True}]


def test_engine_rejects_runtime_precision_drift(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, use_qlora=True)
    member = ScoreMember(
        name="score",
        checkpoint=checkpoint,
        backbone_key="score-backbone",
        parameters_billion=9.0,
        load_in_4bit=True,
    )
    with pytest.raises(ValueError, match="checkpoint use_qlora=False"):
        SubmissionEngine._check_score_precision(
            member, RegressionConfig().with_updates(use_qlora=False)
        )


def test_shared_engine_passes_nf4_to_shared_backbone_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    members = []
    for index in range(2):
        checkpoint = tmp_path / f"checkpoint-{index}"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps({"use_qlora": True}), encoding="utf-8"
        )
        members.append(
            ScoreMember(
                name=f"score-{index}",
                checkpoint=checkpoint,
                backbone_key="shared-backbone",
                parameters_billion=9.0,
                load_in_4bit=True,
            )
        )
    submission = SubmissionConfig(
        name="shared-engine-qlora",
        score_members=tuple(members),
        rationale=RationaleSpec(base_model="stub", adapter=None, enabled=False),
    )
    engine = SubmissionEngine(submission, device="cpu")
    checkpoint_config = RegressionConfig().with_updates(use_qlora=True)
    tokenizer = SimpleNamespace(model_max_length=512)
    calls: list[dict[str, object]] = []

    def fake_shared_loader(paths: list[Path], **kwargs: object) -> SimpleNamespace:
        assert paths == [member.checkpoint for member in members]
        calls.append(kwargs)
        return SimpleNamespace(
            members=[
                SimpleNamespace(
                    config=checkpoint_config,
                    scorer=object(),
                    tokenizer=tokenizer,
                    adapter_name=f"adapter-{index}",
                )
                for index in range(2)
            ]
        )

    monkeypatch.setattr(
        "main_code_submission.engine.load_shared_backbone_checkpoints",
        fake_shared_loader,
    )
    loaded = engine._load_shared_group(members)

    assert len(loaded) == 2
    assert calls[0]["load_in_4bit"] is True
    assert calls[0]["with_lm_head"] is False


def test_docker_runtime_contains_nf4_dependencies_and_model_module() -> None:
    root = Path(__file__).parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    assert "bitsandbytes==0.49.2" in requirements.splitlines()
    assert "main_code/quantization_objectives.py" in dockerfile
