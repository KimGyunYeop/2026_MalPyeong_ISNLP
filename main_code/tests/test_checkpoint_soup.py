"""가중치 수프가 델타의 **정확한** 산술평균인지 검사한다.

왜 이 검사가 핵심인가: LoRA에서 A와 B를 따로 평균하면
``mean(B) @ mean(A) != mean(B @ A)``다. 눈으로는 그럴듯해 보이는 구현이 실제로는
다른 모델을 만든다. 그리고 그 차이는 평가 점수 하나로만 드러나므로, 점수가 나쁘게
나와도 "수프가 원래 안 되는 것"인지 "구현이 틀린 것"인지 구분할 수 없다.
여기서 등식을 직접 확인해 그 모호함을 없앤다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from main_code.checkpoint_soup import SoupError, build_soup, checkpoint_provenance

R = 4
ALPHA = 8
IN_FEATURES = 6
OUT_FEATURES = 5
NAME = "base_model.model.layers.0.self_attn.q_proj"


def _write_member(
    root: Path, index: int, *, seed: int = 43, rank: int = R, alpha: int = ALPHA
) -> Path:
    """resolved_config.json을 가진 run 안에 체크포인트 하나를 만든다."""

    run = root / f"run{index}"
    checkpoint = run / "best_checkpoint"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir(parents=True)
    (checkpoint / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "config.json").write_text(
        json.dumps({"model_id": "test", "pooling": "mean"}), encoding="utf-8"
    )
    (run / "resolved_config.json").write_text(
        json.dumps(
            {"seed": seed, "lora_r": rank, "lora_alpha": alpha, "max_train_steps": 100}
        ),
        encoding="utf-8",
    )
    (checkpoint / "selection.json").write_text(
        json.dumps({"metric": "rmse", "metric_value": 0.4, "global_step": 736 + index}),
        encoding="utf-8",
    )

    generator = torch.Generator().manual_seed(1000 + index)
    save_file(
        {
            f"{NAME}.lora_A.weight": torch.randn(
                rank, IN_FEATURES, generator=generator
            ),
            f"{NAME}.lora_B.weight": torch.randn(
                OUT_FEATURES, rank, generator=generator
            ),
        },
        str(checkpoint / "adapter" / "adapter_model.safetensors"),
    )
    (checkpoint / "adapter" / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": alpha,
                "use_rslora": False,
                "target_modules": ["q_proj"],
                "rank_pattern": {},
                "alpha_pattern": {},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "score_head.weight": torch.full((3, 4), float(index)),
            "step_counter": torch.tensor([index], dtype=torch.long),
        },
        checkpoint / "heads.pt",
    )
    return checkpoint


def _delta(adapter_dir: Path) -> torch.Tensor:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    scaling = config["lora_alpha"] / config["r"]
    return scaling * state[f"{NAME}.lora_B.weight"] @ state[f"{NAME}.lora_A.weight"]


def test_soup_delta_is_the_exact_mean_of_member_deltas(tmp_path: Path) -> None:
    members = [_write_member(tmp_path, index) for index in range(3)]
    build_soup(members, tmp_path / "soup")

    expected = sum(_delta(member / "adapter") for member in members) / len(members)
    observed = _delta(tmp_path / "soup" / "adapter")
    assert torch.allclose(observed, expected, atol=1e-6)


def test_naive_factorwise_average_would_have_been_wrong(tmp_path: Path) -> None:
    """틀린 구현과 결과가 실제로 다름을 고정한다."""

    members = [_write_member(tmp_path, index) for index in range(3)]
    build_soup(members, tmp_path / "soup")

    states = [load_file(str(m / "adapter" / "adapter_model.safetensors")) for m in members]
    naive_a = sum(state[f"{NAME}.lora_A.weight"] for state in states) / len(states)
    naive_b = sum(state[f"{NAME}.lora_B.weight"] for state in states) / len(states)
    naive = (ALPHA / R) * naive_b @ naive_a
    assert not torch.allclose(naive, _delta(tmp_path / "soup" / "adapter"), atol=1e-3)


def test_scaling_ratio_is_preserved(tmp_path: Path) -> None:
    members = [_write_member(tmp_path, index) for index in range(3)]
    build_soup(members, tmp_path / "soup")
    config = json.loads((tmp_path / "soup" / "adapter" / "adapter_config.json").read_text())
    assert config["r"] == R * 3
    assert config["lora_alpha"] == ALPHA * 3
    assert config["lora_alpha"] / config["r"] == ALPHA / R


def test_float_heads_are_averaged_and_integer_buffers_are_not(tmp_path: Path) -> None:
    members = [_write_member(tmp_path, index) for index in range(3)]
    build_soup(members, tmp_path / "soup")
    heads = torch.load(tmp_path / "soup" / "heads.pt", map_location="cpu")
    assert torch.allclose(heads["score_head.weight"], torch.full((3, 4), 1.0))
    # 정수 버퍼를 평균하면 의미 없는 값이 된다. 첫 멤버 값을 그대로 둔다.
    assert heads["step_counter"].tolist() == [0]


def test_manifest_records_every_member(tmp_path: Path) -> None:
    members = [_write_member(tmp_path, index) for index in range(3)]
    manifest = build_soup(members, tmp_path / "soup")
    assert manifest["member_count"] == 3
    assert [item["global_step"] for item in manifest["members"]] == [736, 737, 738]
    assert manifest["delta_semantics"] == "exact_arithmetic_mean_of_member_deltas"
    saved = json.loads((tmp_path / "soup" / "soup_manifest.json").read_text())
    assert saved == manifest


def test_mixed_seeds_are_refused_by_default(tmp_path: Path) -> None:
    members = [
        _write_member(tmp_path, 0, seed=43),
        _write_member(tmp_path, 1, seed=44),
    ]
    with pytest.raises(SoupError, match="다른 분지"):
        build_soup(members, tmp_path / "soup")
    # 명시적으로 허용하면 통과한다. 나쁜 결과가 나와도 원인이 분명해진다.
    build_soup(members, tmp_path / "soup", require_same_seed=False)


def test_mismatched_rank_is_refused(tmp_path: Path) -> None:
    members = [
        _write_member(tmp_path, 0),
        _write_member(tmp_path, 1, rank=R * 2),
    ]
    with pytest.raises(SoupError, match="lora_r"):
        build_soup(members, tmp_path / "soup")


def test_a_single_checkpoint_is_not_a_soup(tmp_path: Path) -> None:
    with pytest.raises(SoupError, match="둘 이상"):
        build_soup([_write_member(tmp_path, 0)], tmp_path / "soup")


def test_rslora_is_refused(tmp_path: Path) -> None:
    members = [_write_member(tmp_path, index) for index in range(2)]
    config_path = members[0] / "adapter" / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["use_rslora"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SoupError, match="rslora"):
        build_soup(members, tmp_path / "soup")


def test_provenance_finds_the_run_configuration(tmp_path: Path) -> None:
    provenance = checkpoint_provenance(_write_member(tmp_path, 0))
    assert provenance["seed"] == 43
    assert provenance["max_train_steps"] == 100
    assert provenance["global_step"] == 736
