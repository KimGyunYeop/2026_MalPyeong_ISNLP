from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from main_code.config import RegressionConfig, load_config, save_config
from main_code.models import _adapter_targets
from main_code.train import LoraJointSchedulerCallback, optimizer_for_training
from main_code.tests.config_helpers import legacy_config


class _LanguageBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.ModuleDict(
            {"q_proj": nn.Linear(4, 4), "o_proj": nn.Linear(4, 4)}
        )
        self.mlp = nn.ModuleDict(
            {"gate_proj": nn.Linear(4, 8), "down_proj": nn.Linear(8, 4)}
        )


class _AdapterBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False)})


class _Scorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _AdapterBackbone()
        self.head = nn.Linear(4, 1)

    def scoring_parameters(self):
        return self.head.parameters()


def _loaded(ratio: float, *, scheduler_scope: str = "global"):
    config = legacy_config(
        training_mode="lora_only" if scheduler_scope == "global" else "two_stage",
        lora_learning_rate=5e-5,
        lora_plus_lr_ratio=ratio,
        lora_scheduler_scope=scheduler_scope,
        max_train_steps=10 if scheduler_scope == "joint" else 0,
        head_warmup_steps=2 if scheduler_scope == "joint" else 0,
    ).validate()
    return SimpleNamespace(scorer=_Scorer(), config=config)


def test_all_linear_targets_every_language_linear_module() -> None:
    model = _LanguageBackbone()
    config = RegressionConfig(lora_targets="all_linear")
    assert _adapter_targets(model, config) == [
        "attention.q_proj",
        "attention.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ]


def test_lora_recipe_config_round_trip(tmp_path) -> None:
    path = tmp_path / "config.json"
    config = RegressionConfig(
        lora_targets="all_linear",
        lora_r=32,
        lora_alpha=64,
        lora_plus_lr_ratio=16.0,
    ).validate()
    save_config(config, path)
    restored = load_config(path)
    assert restored.lora_targets == "all_linear"
    assert restored.lora_r == 32
    assert restored.lora_alpha == 64
    assert restored.lora_plus_lr_ratio == 16.0


def test_default_lora_optimizer_keeps_one_adapter_group() -> None:
    optimizer = optimizer_for_training(_loaded(1.0))
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-5)
    assert "lora_lr_scale" not in optimizer.param_groups[1]


def test_lora_plus_uses_b_learning_rate_ratio() -> None:
    optimizer = optimizer_for_training(_loaded(16.0))
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5e-5)
    assert optimizer.param_groups[2]["lr"] == pytest.approx(8e-4)
    assert optimizer.param_groups[1]["lora_lr_scale"] == 1.0
    assert optimizer.param_groups[2]["lora_lr_scale"] == 16.0


def test_joint_scheduler_preserves_lora_plus_ratio() -> None:
    loaded = _loaded(16.0, scheduler_scope="joint")
    optimizer = optimizer_for_training(loaded)
    callback = LoraJointSchedulerCallback(loaded.scorer, optimizer, loaded.config)
    state = SimpleNamespace(global_step=2, max_steps=10, epoch=0)
    callback.on_step_begin(None, state, None)
    assert optimizer.param_groups[2]["lr"] == pytest.approx(
        optimizer.param_groups[1]["lr"] * 16.0
    )
    assert callback.summary()["group_lr_scales"] == [1.0, 16.0]
