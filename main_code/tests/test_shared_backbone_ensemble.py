"""공유 backbone 다중 어댑터가 독립 로드와 **같은 점수**를 내는지 검증한다.

이 test가 없으면 어댑터 공유는 쓸 수 없다. VRAM을 아끼려고 점수가 조용히 달라지면
offline validation 점수와 제출 점수가 갈라지고, 이 프로젝트는 이미 같은 종류의 불일치로
organization RMSE가 `.54 -> 1.03`으로 무너진 전례가 있다.

무작위 초기화한 아주 작은 decoder로 CPU에서 돌린다. 실제 7B가 아니어도 검증 대상인
"어댑터 스왑 경로가 독립 로드와 동일한가"는 그대로 확인된다.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from main_code.config import RegressionConfig, save_config  # noqa: E402
from main_code.datasets import EssayRegressionDataset, RegressionCollator  # noqa: E402
from main_code.models import (  # noqa: E402
    build_model,
    load_checkpoint,
    load_shared_backbone_checkpoints,
)

ROW = {
    "id": "test-1",
    "prompt": "로봇세 도입에 대한 의견을 쓰시오.",
    "essay": "로봇세는 필요하다. 자동화가 일자리를 줄이기 때문이다. 따라서 재원이 필요하다.",
}


def _tiny_backbone(directory: Path) -> str:
    """무작위 초기화한 2층 Llama와 tokenizer를 로컬에 저장한다."""

    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    directory.mkdir(parents=True, exist_ok=True)
    corpus = [ROW["prompt"], ROW["essay"], "점수 채점 내용 구성 표현"]
    raw = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator(
        corpus,
        trainers.WordPieceTrainer(
            vocab_size=200, special_tokens=["[UNK]", "[PAD]", "[EOS]"]
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


def _make_checkpoint(
    destination: Path,
    model_path: str,
    *,
    seed: int,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_include_mlp: bool = True,
) -> RegressionConfig:
    """작은 LoRA checkpoint 하나를 실제 저장 형식으로 만든다."""

    config = RegressionConfig(
        model_id=model_path,
        model_revision="main",
        trust_remote_code=False,
        backbone_type="decoder",
        training_mode="lora_only",
        essay_surface="official_raw",
        max_length=128,
        batch_size=2,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_include_mlp=lora_include_mlp,
        seed=seed,
    ).validate()
    torch.manual_seed(seed)
    loaded = build_model(config, device=torch.device("cpu"))
    # 어댑터와 head에 서로 다른 값을 넣어 구성원이 실제로 다른 점수를 내게 한다.
    with torch.no_grad():
        for name, parameter in loaded.scorer.named_parameters():
            if "lora_B" in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
            elif not name.startswith("backbone."):
                parameter.add_(torch.randn_like(parameter) * 0.02)

    destination.mkdir(parents=True, exist_ok=True)
    loaded.scorer.backbone.save_pretrained(str(destination / "adapter"))
    scoring_state = {
        key: value
        for key, value in loaded.scorer.state_dict().items()
        if not key.startswith("backbone.")
    }
    torch.save(
        {"schema_version": 2, "scoring_state": scoring_state},
        destination / "heads.pt",
    )
    loaded.tokenizer.save_pretrained(str(destination / "tokenizer"))
    save_config(config, destination / "config.json")
    return config


def _make_rationale_adapter(
    destination: Path,
    model_path: str,
    *,
    lora_r: int = 2,
    lora_alpha: int = 4,
    target_modules: list[str] | None = None,
) -> Path:
    """Save the same namespace shape as the real CAUSAL_LM rationale LoRA."""

    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path)
    adapter = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=(
                target_modules
                if target_modules is not None
                else ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
        ),
    )
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    adapter.save_pretrained(destination)
    return destination


def _score(scorer, tokenizer, config) -> list[float]:
    dataset = EssayRegressionDataset(
        [dict(ROW)], config, split="inference", require_labels=False
    )
    collator = RegressionCollator(
        tokenizer, config, include_labels=False, max_length=config.max_length
    )
    batch = collator([dataset[0]])
    inputs = {
        key: value
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key != "labels"
    }
    with torch.no_grad():
        result = scorer(
            **inputs,
            return_probabilities=(config.score_head == "distribution"),
            return_detail_predictions=(config.detail_head_mode != "none"),
        )
    return result["scores"].detach().float().reshape(-1).tolist()


@pytest.fixture(scope="module")
def two_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("shared_backbone")
    model_path = _tiny_backbone(root / "tiny_model")
    first = root / "member_a"
    second = root / "member_b"
    # Expanded score shape versus historical score shape.  PEFT must preserve
    # each adapter's own rank/targets on one backbone.
    _make_checkpoint(
        first,
        model_path,
        seed=42,
        lora_r=4,
        lora_alpha=8,
        lora_include_mlp=True,
    )
    _make_checkpoint(
        second,
        model_path,
        seed=43,
        lora_r=2,
        lora_alpha=4,
        lora_include_mlp=False,
    )
    return first, second


def test_shared_scores_match_independent_loads_exactly(
    two_checkpoints: tuple[Path, Path],
) -> None:
    """어댑터 스왑으로 낸 점수가 독립 로드 점수와 정확히 같아야 한다."""

    first, second = two_checkpoints
    device = torch.device("cpu")

    independent: list[list[float]] = []
    for path in (first, second):
        loaded = load_checkpoint(path, device=device)
        independent.append(_score(loaded.scorer, loaded.tokenizer, loaded.config))
        del loaded

    ensemble = load_shared_backbone_checkpoints(
        [first, second], device=device, names=["a", "b"]
    )
    assert len(ensemble.members) == 2
    assert ensemble.adapter_names == ["default", "member_1"]
    expanded = ensemble.backbone.peft_config["default"]
    historical = ensemble.backbone.peft_config["member_1"]
    assert (expanded.r, expanded.lora_alpha) == (4, 8)
    assert (historical.r, historical.lora_alpha) == (2, 4)
    assert {"gate_proj", "up_proj", "down_proj"}.issubset(
        set(expanded.target_modules)
    )
    assert set(historical.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }

    shared: list[list[float]] = []
    for member in ensemble.members:
        ensemble.activate(member.adapter_name)
        shared.append(_score(member.scorer, member.tokenizer, member.config))

    for name, want, got in zip(("a", "b"), independent, shared, strict=True):
        for index, (expected, actual) in enumerate(zip(want, got, strict=True)):
            assert actual == pytest.approx(expected, abs=1e-6), (
                f"{name} trait{index}: 공유 경로 {actual} != 독립 로드 {expected}"
            )

    # 두 구성원이 실제로 다른 점수를 내야 이 test가 의미가 있다.
    assert shared[0] != shared[1]


def test_activation_is_required_and_order_independent(
    two_checkpoints: tuple[Path, Path],
) -> None:
    """어댑터를 바꾸면 점수가 바뀌고, 되돌리면 원래 점수로 정확히 돌아와야 한다."""

    first, second = two_checkpoints
    ensemble = load_shared_backbone_checkpoints(
        [first, second], device=torch.device("cpu"), names=["a", "b"]
    )
    member_a, member_b = ensemble.members

    ensemble.activate(member_a.adapter_name)
    a_first = _score(member_a.scorer, member_a.tokenizer, member_a.config)
    ensemble.activate(member_b.adapter_name)
    _score(member_b.scorer, member_b.tokenizer, member_b.config)
    ensemble.activate(member_a.adapter_name)
    a_again = _score(member_a.scorer, member_a.tokenizer, member_a.config)
    assert a_again == pytest.approx(a_first, abs=1e-9)

    # using()은 끝나면 이전 어댑터로 되돌린다.
    with ensemble.using(member_b.adapter_name):
        assert ensemble._active == member_b.adapter_name
    assert ensemble._active == member_a.adapter_name


def test_wrong_adapter_changes_scores(two_checkpoints: tuple[Path, Path]) -> None:
    """활성 어댑터가 틀리면 점수가 달라진다.

    즉 `activate` 호출을 빼먹으면 조용히 다른 모델의 점수가 나온다. 그래서 engine은 멤버마다
    반드시 activate를 호출해야 하고, 이 test가 그 필요성을 고정한다.
    """

    first, second = two_checkpoints
    ensemble = load_shared_backbone_checkpoints(
        [first, second], device=torch.device("cpu"), names=["a", "b"]
    )
    member_a, member_b = ensemble.members

    ensemble.activate(member_a.adapter_name)
    correct = _score(member_b.scorer, member_b.tokenizer, member_b.config)
    ensemble.activate(member_b.adapter_name)
    proper = _score(member_b.scorer, member_b.tokenizer, member_b.config)
    assert correct != pytest.approx(proper, abs=1e-6)


def test_causal_lm_only_score_parity_and_rationale_adapter_switch(
    two_checkpoints: tuple[Path, Path],
) -> None:
    """One retained CausalLM must serve score hidden states and rationale LoRA.

    This covers the exact artifact namespace mismatch in the real submission:
    FEATURE_EXTRACTION score keys have one ``model`` level and CAUSAL_LM rationale
    keys have two. Loading must be strict, score parity must hold, and leaving the
    generation context must restore the score adapter.
    """

    first, _ = two_checkpoints
    device = torch.device("cpu")
    independent = load_checkpoint(first, device=device)
    expected = _score(
        independent.scorer,
        independent.tokenizer,
        independent.config,
    )

    ensemble = load_shared_backbone_checkpoints(
        [first],
        device=device,
        names=["score"],
        with_lm_head=True,
    )
    assert ensemble.causal_lm is not None
    member = ensemble.members[0]
    assert _score(member.scorer, member.tokenizer, member.config) == pytest.approx(
        expected, abs=1e-6
    )

    rationale = _make_rationale_adapter(
        first.parent / "rationale_adapter",
        member.config.model_id,
    )
    rationale_name = ensemble.load_causal_adapter(rationale)
    assert rationale_name == "rationale"
    score_peft = ensemble.backbone.peft_config[member.adapter_name]
    rationale_peft = ensemble.backbone.peft_config[rationale_name]
    assert (score_peft.r, score_peft.lora_alpha) == (4, 8)
    assert (rationale_peft.r, rationale_peft.lora_alpha) == (2, 4)
    assert set(rationale_peft.target_modules) < set(score_peft.target_modules)
    q_proj = next(
        module
        for name, module in ensemble.backbone.named_modules()
        if name.endswith("q_proj") and hasattr(module, "lora_A")
    )
    gate_proj = next(
        module
        for name, module in ensemble.backbone.named_modules()
        if name.endswith("gate_proj") and hasattr(module, "lora_A")
    )
    assert tuple(q_proj.lora_A[member.adapter_name].weight.shape) == (4, 32)
    assert tuple(q_proj.lora_A[rationale_name].weight.shape) == (2, 32)
    assert rationale_name not in gate_proj.lora_A

    input_ids = torch.tensor([[1, 5, 6]], dtype=torch.long)
    ensemble.activate(member.adapter_name)
    score_logits = ensemble.causal_lm(input_ids=input_ids).logits.detach().clone()
    with ensemble.using(rationale_name):
        rationale_logits = (
            ensemble.causal_lm(input_ids=input_ids).logits.detach().clone()
        )
        generated = ensemble.causal_lm.generate(
            input_ids=input_ids,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=2,
        )
        assert generated.shape[1] >= input_ids.shape[1]
    assert ensemble._active == member.adapter_name
    assert not torch.allclose(score_logits, rationale_logits)
    assert _score(member.scorer, member.tokenizer, member.config) == pytest.approx(
        expected, abs=1e-6
    )


def test_causal_adapter_contract_rejects_non_causal_task(
    two_checkpoints: tuple[Path, Path],
) -> None:
    first, _ = two_checkpoints
    ensemble = load_shared_backbone_checkpoints(
        [first],
        device=torch.device("cpu"),
        names=["score"],
        with_lm_head=True,
    )
    source = _make_rationale_adapter(
        first.parent / "rationale_adapter_bad_source",
        ensemble.members[0].config.model_id,
    )
    broken = first.parent / "rationale_adapter_bad"
    shutil.copytree(source, broken)
    config_path = broken / "adapter_config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["task_type"] = "FEATURE_EXTRACTION"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="CAUSAL_LM LoRA"):
        ensemble.load_causal_adapter(broken)


def _corrupt_adapter_weights(source: Path, destination: Path, mode: str) -> Path:
    from safetensors.torch import load_file, save_file

    shutil.copytree(source, destination)
    weights_path = destination / "adapter_model.safetensors"
    state = load_file(str(weights_path))
    if mode == "missing":
        state.pop(next(iter(state)))
    elif mode == "unexpected":
        example = next(iter(state.values()))
        state["base_model.model.model.not_a_real_layer.lora_A.weight"] = (
            torch.zeros_like(example)
        )
    else:  # pragma: no cover - test helper contract
        raise ValueError(mode)
    save_file(state, str(weights_path), metadata={"format": "pt"})
    return destination


@pytest.mark.parametrize("mode", ["missing", "unexpected"])
def test_first_score_adapter_tensor_mismatch_fails_closed(
    two_checkpoints: tuple[Path, Path],
    tmp_path: Path,
    mode: str,
) -> None:
    """첫 score adapter도 PEFT 경고로 계속 가지 않고 즉시 실패해야 한다."""

    first, _ = two_checkpoints
    broken = tmp_path / f"broken_first_{mode}"
    shutil.copytree(first, broken)
    _corrupt_adapter_weights(first / "adapter", broken / "adapter_broken", mode)
    shutil.rmtree(broken / "adapter")
    (broken / "adapter_broken").rename(broken / "adapter")

    with pytest.raises(ValueError, match=rf"{mode}="):
        load_checkpoint(broken, device=torch.device("cpu"))


@pytest.mark.parametrize("mode", ["missing", "unexpected"])
def test_causal_adapter_tensor_mismatch_fails_closed_and_rolls_back(
    two_checkpoints: tuple[Path, Path],
    tmp_path: Path,
    mode: str,
) -> None:
    first, _ = two_checkpoints
    ensemble = load_shared_backbone_checkpoints(
        [first], device=torch.device("cpu"), names=["score"], with_lm_head=True
    )
    member = ensemble.members[0]
    ensemble.activate(member.adapter_name)
    expected = _score(member.scorer, member.tokenizer, member.config)
    source = _make_rationale_adapter(tmp_path / f"source_{mode}", member.config.model_id)
    broken = _corrupt_adapter_weights(source, tmp_path / f"broken_{mode}", mode)

    with pytest.raises(ValueError, match=rf"{mode}="):
        ensemble.load_causal_adapter(broken)

    assert "rationale" not in ensemble.backbone.peft_config
    assert ensemble.active_adapter == member.adapter_name
    assert _score(member.scorer, member.tokenizer, member.config) == pytest.approx(
        expected, abs=1e-6
    )


def test_causal_adapter_rejects_targets_outside_score_adapter(
    two_checkpoints: tuple[Path, Path], tmp_path: Path
) -> None:
    first, _ = two_checkpoints
    ensemble = load_shared_backbone_checkpoints(
        [first], device=torch.device("cpu"), names=["score"], with_lm_head=True
    )
    source = _make_rationale_adapter(
        tmp_path / "rationale_target_source",
        ensemble.members[0].config.model_id,
    )
    broken = tmp_path / "rationale_target_outside_score"
    shutil.copytree(source, broken)
    config_path = broken / "adapter_config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["target_modules"].append("embed_tokens")
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="subset"):
        ensemble.load_causal_adapter(broken)
    assert "rationale" not in ensemble.backbone.peft_config


def test_mismatched_backbone_fails_closed(tmp_path: Path) -> None:
    """backbone이 다른 checkpoint를 공유하려 하면 즉시 중단해야 한다."""

    model_a = _tiny_backbone(tmp_path / "model_a")
    model_b = _tiny_backbone(tmp_path / "model_b")
    first = tmp_path / "a"
    second = tmp_path / "b"
    _make_checkpoint(first, model_a, seed=42)
    _make_checkpoint(second, model_b, seed=42)
    with pytest.raises(ValueError, match="backbone identity가 다른"):
        load_shared_backbone_checkpoints(
            [first, second], device=torch.device("cpu"), names=["a", "b"]
        )


def test_duplicate_member_names_fail_closed(
    two_checkpoints: tuple[Path, Path],
) -> None:
    first, second = two_checkpoints
    with pytest.raises(ValueError, match="중복"):
        load_shared_backbone_checkpoints(
            [first, second], device=torch.device("cpu"), names=["a", "a"]
        )
