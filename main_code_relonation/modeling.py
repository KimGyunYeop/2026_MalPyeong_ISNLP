from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .config import RationaleConfig


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _template_kwargs(config: RationaleConfig) -> dict[str, Any]:
    return dict(config.chat_template_kwargs or {})


def apply_chat_template_ids(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool,
    config: RationaleConfig,
) -> list[int]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer에 chat_template이 없습니다")
    value = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **_template_kwargs(config),
    )
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.squeeze(0).tolist()
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("apply_chat_template가 token ID list를 반환하지 않았습니다")
    return value


def assistant_supervision_ids(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    assistant_target: str,
    *,
    config: RationaleConfig,
) -> tuple[list[int], list[int], int]:
    prompt_ids = apply_chat_template_ids(
        tokenizer, messages, add_generation_prompt=True, config=config
    )
    full_messages = [*messages, {"role": "assistant", "content": assistant_target}]
    full_ids = apply_chat_template_ids(
        tokenizer, full_messages, add_generation_prompt=False, config=config
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(prompt_ids, full_ids, strict=False)
                )
                if left != right
            ),
            min(len(prompt_ids), len(full_ids)),
        )
        raise ValueError(
            "chat template의 generation prompt가 assistant full message prefix와 "
            f"다릅니다: token_index={mismatch}"
        )
    target_length = len(full_ids) - len(prompt_ids)
    if target_length < 1:
        raise ValueError("assistant target token이 없습니다")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return full_ids, labels, len(prompt_ids)


def load_tokenizer(config: RationaleConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer에 pad/eos token이 모두 없습니다")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: RationaleConfig, *, for_training: bool) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    set_seed(config.seed)

    quantization = None
    if config.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype(config.torch_dtype),
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
        torch_dtype=(None if config.load_in_4bit else torch_dtype(config.torch_dtype)),
        quantization_config=quantization,
        device_map="auto",
    )
    if for_training:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_targets),
                bias="none",
            ),
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    elif config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, config.adapter_path)
    if for_training:
        model.train()
    else:
        model.eval()
    return model


@dataclass(frozen=True)
class GenerationResult:
    raw: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


class GeneratorRuntime:
    def __init__(self, config: RationaleConfig) -> None:
        config.validate()
        self.config = config
        self.tokenizer = load_tokenizer(config)
        self.model = load_model(config, for_training=False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        prompt_ids = apply_chat_template_ids(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
            config=self.config,
        )
        requested = max_new_tokens or self.config.max_new_tokens
        requested = int(requested)
        if not 1 <= requested <= 2048:
            raise ValueError("max_new_tokens는 1~2048이어야 합니다")
        if len(prompt_ids) + requested > self.config.max_length:
            raise ValueError(
                "prompt+completion이 max_length를 넘습니다: "
                f"{len(prompt_ids)}+{requested}>{self.config.max_length}"
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        kwargs: dict[str, Any] = {
            "max_new_tokens": requested,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.config.temperature > 0:
            kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        started = time.perf_counter()
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        latency = time.perf_counter() - started
        completion_ids = output[0, len(prompt_ids) :].tolist()
        raw = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerationResult(raw, len(prompt_ids), len(completion_ids), latency)
