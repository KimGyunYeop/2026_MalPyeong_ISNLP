from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from main_code.config import RegressionConfig, load_config, save_config
from main_code.models import _load_trainable_adapter, _prepare_for_adapter_training
from main_code.utils import resolve_text_model


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_enabled = False
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True


def test_initial_adapter_config_round_trip_and_mode_guard(tmp_path) -> None:
    path = tmp_path / "config.json"
    config = RegressionConfig(
        training_mode="lora_only",
        initial_lora_adapter="/tmp/preadapt/checkpoint/adapter",
    ).validate()
    save_config(config, path)
    assert load_config(path).initial_lora_adapter == config.initial_lora_adapter

    with pytest.raises(ValueError, match="lora_only 또는 two_stage"):
        RegressionConfig(
            training_mode="head_only",
            initial_lora_adapter="/tmp/preadapt/checkpoint/adapter",
        ).validate()


def test_initial_adapter_is_loaded_trainable_with_checkpointing(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _PeftModel:
        @staticmethod
        def from_pretrained(model, path, *, is_trainable):
            calls.update(model=model, path=path, is_trainable=is_trainable)
            return model

    peft = ModuleType("peft")
    peft.PeftModel = _PeftModel
    peft.prepare_model_for_kbit_training = lambda model, **_: model
    monkeypatch.setitem(sys.modules, "peft", peft)

    backbone = _Backbone()
    config = RegressionConfig(
        training_mode="lora_only",
        initial_lora_adapter="/tmp/preadapt/checkpoint/adapter",
        gradient_checkpointing=True,
    ).validate()
    loaded = _load_trainable_adapter(backbone, config)

    assert loaded is backbone
    assert calls["path"] == config.initial_lora_adapter
    assert calls["is_trainable"] is True
    assert backbone.gradient_checkpointing_enabled
    assert backbone.input_grads_enabled
    assert backbone.config.use_cache is False


class _HookBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_calls = 0
        self.input_grad_calls = 0

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_calls += 1
        self.gradient_checkpointing = True

    def enable_input_require_grads(self) -> None:
        self.input_grad_calls += 1

        def require_grad(_module, _inputs, output):
            output.requires_grad_(True)

        self._require_grads_hook = self.embedding.register_forward_hook(require_grad)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


def _install_prepare(
    monkeypatch, prepare
) -> None:
    peft = ModuleType("peft")
    peft.prepare_model_for_kbit_training = prepare
    monkeypatch.setitem(sys.modules, "peft", peft)


@pytest.mark.parametrize("wrapper", ("qwen", "gemma"))
def test_marker_lost_multimodal_qlora_child_enables_gc_and_input_grads(
    monkeypatch, wrapper: str
) -> None:
    child = _HookBackbone()
    container = nn.Module()
    container.is_loaded_in_4bit = True
    if wrapper == "qwen":
        container.language_model = child
    else:
        inner = nn.Module()
        inner.language_model = child
        container.model = inner
    assert resolve_text_model(container) is child
    assert not hasattr(child, "is_loaded_in_4bit")

    observed: dict[str, object] = {}

    def prepare(model, *, use_gradient_checkpointing):
        observed["marker"] = getattr(model, "is_loaded_in_4bit", False)
        observed["use_gradient_checkpointing"] = use_gradient_checkpointing
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # Reproduce the PEFT bug: marker-less child returns without enabling GC.
        return model

    _install_prepare(monkeypatch, prepare)
    config = RegressionConfig(
        training_mode="lora_only",
        use_qlora=True,
        gradient_checkpointing=True,
    ).validate()
    prepared = _prepare_for_adapter_training(child, config)

    assert prepared is child
    assert observed == {"marker": False, "use_gradient_checkpointing": True}
    assert child.is_gradient_checkpointing
    assert child.gradient_checkpointing_calls == 1
    assert child.input_grad_calls == 1
    assert child.config.use_cache is False
    output = child(torch.tensor([[1, 2]]))
    assert output.requires_grad


def test_ordinary_marked_qlora_does_not_duplicate_existing_gc_hooks(
    monkeypatch,
) -> None:
    backbone = _HookBackbone()
    backbone.is_loaded_in_4bit = True

    def prepare(model, *, use_gradient_checkpointing):
        assert use_gradient_checkpointing is True
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        return model

    _install_prepare(monkeypatch, prepare)
    config = RegressionConfig(
        training_mode="lora_only",
        use_qlora=True,
        gradient_checkpointing=True,
    ).validate()
    _prepare_for_adapter_training(backbone, config)

    assert backbone.gradient_checkpointing_calls == 1
    assert backbone.input_grad_calls == 1
    assert backbone.is_gradient_checkpointing


def test_qlora_no_gradient_checkpointing_remains_an_activation_noop(
    monkeypatch,
) -> None:
    backbone = _HookBackbone()
    backbone.is_loaded_in_4bit = True
    observed: dict[str, object] = {}

    def prepare(model, *, use_gradient_checkpointing):
        observed["use_gradient_checkpointing"] = use_gradient_checkpointing
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    _install_prepare(monkeypatch, prepare)
    config = RegressionConfig(
        training_mode="lora_only",
        use_qlora=True,
        gradient_checkpointing=False,
    ).validate()
    _prepare_for_adapter_training(backbone, config)

    assert observed["use_gradient_checkpointing"] is False
    assert not backbone.is_gradient_checkpointing
    assert backbone.gradient_checkpointing_calls == 0
    assert backbone.input_grad_calls == 0
    assert not backbone(torch.tensor([[1, 2]])).requires_grad
    assert backbone.config.use_cache is False


def test_gradient_checkpointing_fail_closed_when_backend_does_not_activate(
    monkeypatch,
) -> None:
    class Broken(_HookBackbone):
        def gradient_checkpointing_enable(self) -> None:
            self.gradient_checkpointing_calls += 1

        def enable_input_require_grads(self) -> None:
            self.input_grad_calls += 1

    backbone = Broken()
    _install_prepare(
        monkeypatch,
        lambda model, *, use_gradient_checkpointing: model,
    )
    config = RegressionConfig(
        training_mode="lora_only",
        use_qlora=True,
        gradient_checkpointing=True,
    ).validate()
    with pytest.raises(RuntimeError, match="활성화 검증 실패"):
        _prepare_for_adapter_training(backbone, config)
