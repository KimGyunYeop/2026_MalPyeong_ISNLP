from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .artifacts import sha256_text
from .prompts import (
    DEFAULT_SKELETON_HINT,
    SUBMITTED_SKELETON_HINT,
    baseline_prompt_template,
    submitted_prompt_template,
    load_prompt_template,
    prompt_template_id,
    prompt_template_sha256,
    validate_prompt_template,
)


@dataclass(frozen=True)
class RationaleConfig:
    # 2026-08-25 최종 제출본 기본값. 인자 없이 실행하면 제출본 근거모델을 재현한다.
    model_id: str = "Qwen/Qwen3.5-9B"
    model_revision: str = "main"
    trust_remote_code: bool = True
    adapter_path: str | None = None
    load_in_4bit: bool = False
    torch_dtype: str = "bfloat16"
    max_length: int = 8192
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    score_mode: str = "fixed"
    # Exact training/inference prompt is part of the model artifact.  Old recipes
    # omit these fields and therefore resolve to the byte-preserved baseline.
    rationale_prompt_id: str = "baseline_prompt"
    rationale_prompt_text: str = baseline_prompt_template()
    rationale_prompt_sha256: str = prompt_template_sha256(baseline_prompt_template())
    rationale_prompt_source: str = "baseline_fallback"
    # 출력 스켈레톤의 rationale 자리표시자에 들어가는 길이 지시. 스켈레톤은 생성 직전에
    # 붙어 본문 규칙보다 강하게 작동하므로, 프롬프트 판본이 다른 길이를 요구하면 이
    # 값도 함께 바꿔야 한다. 기본값은 역사적 문자열이고 `to_dict`에서 기본값일 때
    # 생략하므로 기존 recipe의 fingerprint와 렌더 결과가 그대로 유지된다.
    rationale_skeleton_hint: str = DEFAULT_SKELETON_HINT
    chat_template_kwargs: dict[str, Any] | None = field(
        default_factory=lambda: {"enable_thinking": False}
    )
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    batch_size: int = 2
    gradient_accumulation: int = 16
    learning_rate: float = 4e-5
    epochs: float = 2.0
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0

    def validate(self) -> None:
        if not self.model_id:
            raise ValueError("model_id가 필요합니다")
        if self.score_mode not in {"fixed", "joint"}:
            raise ValueError("score_mode은 fixed 또는 joint여야 합니다")
        validate_prompt_template(self.rationale_prompt_text)
        actual_prompt_hash = prompt_template_sha256(self.rationale_prompt_text)
        if self.rationale_prompt_sha256 != actual_prompt_hash:
            raise ValueError(
                "rationale prompt 원문과 SHA-256이 다릅니다: "
                f"expected={self.rationale_prompt_sha256} actual={actual_prompt_hash}"
            )
        if not self.rationale_prompt_id or not self.rationale_prompt_source:
            raise ValueError("rationale prompt id/source가 필요합니다")
        if self.torch_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("지원하지 않는 torch_dtype입니다")
        if self.max_length < 1 or not 1 <= self.max_new_tokens <= 2048:
            raise ValueError("max_length/max_new_tokens가 올바르지 않습니다")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature/top_p가 올바르지 않습니다")
        if self.lora_rank < 1 or self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("LoRA/batch 설정은 양수여야 합니다")
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError("learning_rate/epochs는 양수여야 합니다")

    def with_updates(self, **values: Any) -> "RationaleConfig":
        result = replace(self, **values)
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lora_targets"] = list(self.lora_targets)
        # 기본 hint는 직렬화하지 않는다. 이 필드를 추가하기 전에 학습한 recipe의
        # fingerprint(r12=bdf21a4b13f3, r14=0326cc92e4d0)와 그 결과 leaf 이름을
        # 그대로 재현하기 위한 의도적 생략이다. 값을 바꾼 recipe에만 나타난다.
        if value.get("rationale_skeleton_hint") == DEFAULT_SKELETON_HINT:
            value.pop("rationale_skeleton_hint", None)
        return value

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return sha256_text(payload)[:12]


def load_config(path: str | Path) -> RationaleConfig:
    recipe_path = Path(path).resolve()
    value = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recipe는 JSON 객체여야 합니다")
    prompt_file = value.pop("rationale_prompt_file", None)
    embedded_prompt = value.get("rationale_prompt_text")
    prompt_metadata_keys = {
        "rationale_prompt_id",
        "rationale_prompt_sha256",
        "rationale_prompt_source",
    }
    declared_prompt_metadata = prompt_metadata_keys.intersection(value)
    if prompt_file is not None and embedded_prompt is not None:
        raise ValueError(
            "recipe에는 rationale_prompt_file 또는 embedded text 중 하나만 넣으세요"
        )
    if prompt_file is not None and declared_prompt_metadata:
        raise ValueError(
            "rationale_prompt_file을 쓰면 prompt id/SHA/source를 중복 선언하지 마세요"
        )
    if prompt_file is None and embedded_prompt is None and declared_prompt_metadata:
        raise ValueError(
            "rationale prompt 메타데이터만 일부 선언할 수 없습니다. "
            "원문 또는 rationale_prompt_file을 함께 지정하세요"
        )
    if prompt_file is not None:
        source_path = Path(str(prompt_file)).expanduser()
        if not source_path.is_absolute():
            source_path = recipe_path.parent / source_path
        source_path = source_path.resolve()
        template = load_prompt_template(source_path)
        value.update(
            {
                "rationale_prompt_id": prompt_template_id(template),
                "rationale_prompt_text": template,
                "rationale_prompt_sha256": prompt_template_sha256(template),
                "rationale_prompt_source": f"file:{source_path}",
            }
        )
    elif embedded_prompt is not None:
        template = validate_prompt_template(str(embedded_prompt))
        actual_hash = prompt_template_sha256(template)
        actual_id = prompt_template_id(template)
        declared_hash = value.get("rationale_prompt_sha256", actual_hash)
        if declared_hash != actual_hash:
            raise ValueError(
                "embedded rationale prompt SHA-256 불일치: "
                f"expected={declared_hash} actual={actual_hash}"
            )
        declared_id = value.get("rationale_prompt_id", actual_id)
        if declared_id != actual_id:
            raise ValueError(
                "embedded rationale prompt ID가 원문과 다릅니다: "
                f"expected={declared_id} actual={actual_id}"
            )
        value["rationale_prompt_sha256"] = actual_hash
        value["rationale_prompt_id"] = actual_id
        value.setdefault("rationale_prompt_source", "embedded")
    if "lora_targets" in value:
        value["lora_targets"] = tuple(value["lora_targets"])
    config = RationaleConfig(**value)
    config.validate()
    return config


def with_prompt_file(
    config: RationaleConfig, path: str | Path, *, source_label: str = "cli"
) -> RationaleConfig:
    source_path = Path(path).expanduser().resolve()
    template = load_prompt_template(source_path)
    return config.with_updates(
        rationale_prompt_id=prompt_template_id(template),
        rationale_prompt_text=template,
        rationale_prompt_sha256=prompt_template_sha256(template),
        rationale_prompt_source=f"{source_label}:{source_path}",
    )


def bind_adapter_prompt(
    config: RationaleConfig,
    adapter_path: str | Path,
    *,
    prompt_was_explicit: bool,
) -> RationaleConfig:
    """Bind inference to the exact prompt saved beside a trained adapter.

    Legacy adapters have no sidecar and are valid only with the preserved baseline.
    New adapters must carry both the text and a small runtime metadata JSON.
    """

    adapter = Path(adapter_path).expanduser().resolve()
    prompt_path = adapter / "rationale_prompt.txt"
    runtime_path = adapter / "rationale_runtime_config.json"
    if not prompt_path.exists() and not runtime_path.exists():
        if (
            prompt_was_explicit
            or config.rationale_prompt_text != baseline_prompt_template()
        ):
            raise ValueError(
                "prompt sidecar가 없는 legacy adapter에는 baseline prompt만 사용할 수 있습니다: "
                f"{adapter}"
            )
        return config
    if not prompt_path.is_file() or not runtime_path.is_file():
        raise FileNotFoundError(
            "rationale adapter prompt sidecar가 불완전합니다: "
            f"text={prompt_path.is_file()} config={runtime_path.is_file()}"
        )
    template = validate_prompt_template(prompt_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(runtime, dict):
        raise ValueError("rationale_runtime_config.json은 JSON object여야 합니다")
    digest = prompt_template_sha256(template)
    declared_text = runtime.get("rationale_prompt_text", runtime.get("prompt_text"))
    declared_hash = runtime.get("rationale_prompt_sha256", runtime.get("prompt_sha256"))
    if declared_text != template or declared_hash != digest:
        raise ValueError("rationale adapter prompt sidecar 원문/SHA가 서로 다릅니다")
    if prompt_was_explicit and (
        config.rationale_prompt_text != template
        or config.rationale_prompt_sha256 != digest
    ):
        raise ValueError(
            "명시한 rationale prompt가 adapter 학습 prompt와 다릅니다: "
            f"config={config.rationale_prompt_sha256} adapter={digest}"
        )
    return config.with_updates(
        rationale_prompt_id=str(
            runtime.get("rationale_prompt_id", runtime.get("prompt_id"))
            or prompt_template_id(template)
        ),
        rationale_prompt_text=template,
        rationale_prompt_sha256=digest,
        rationale_prompt_source=f"adapter:{adapter}",
    )
