"""Bind a rationale adapter to the exact prompt template used for training.

The prompt is a model artifact, not a serving-time preference.  New rationale
adapters therefore carry both ``rationale_prompt.txt`` and
``rationale_runtime_config.json``.  Historical adapters predate those sidecars;
their one intentional compatibility rule is the immutable baseline template.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from main_code_relonation.prompts import (
    baseline_prompt_template,
    prompt_template_id,
    prompt_template_sha256,
)


PROMPT_TEXT_FILENAME = "rationale_prompt.txt"
RUNTIME_CONFIG_FILENAME = "rationale_runtime_config.json"
MANIFEST_PROMPT_KEYS = (
    "rationale_prompt_id",
    "rationale_prompt_text",
    "rationale_prompt_sha256",
)


@dataclass(frozen=True)
class RationalePromptBinding:
    prompt_id: str
    text: str
    sha256: str

    def manifest_fields(self) -> dict[str, str]:
        return {
            "rationale_prompt_id": self.prompt_id,
            "rationale_prompt_text": self.text,
            "rationale_prompt_sha256": self.sha256,
        }


def validate_prompt_binding(
    prompt_id: Any,
    text: Any,
    sha256: Any,
    *,
    source: str,
) -> RationalePromptBinding:
    """Validate an exact prompt-text/hash triple and return its canonical form."""

    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError(f"{source}: rationale_prompt_id가 비었습니다")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{source}: rationale_prompt_text가 비었습니다")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(
            f"{source}: rationale_prompt_sha256가 유효한 SHA-256이 아닙니다"
        )
    actual = prompt_template_sha256(text)
    if actual != sha256:
        raise ValueError(
            f"{source}: rationale prompt 해시가 원문과 다릅니다: "
            f"expected={sha256} got={actual}"
        )
    return RationalePromptBinding(prompt_id.strip(), text, sha256)


def baseline_prompt_binding() -> RationalePromptBinding:
    text = baseline_prompt_template()
    return validate_prompt_binding(
        prompt_template_id(text),
        text,
        prompt_template_sha256(text),
        source="baseline rationale prompt",
    )


def prompt_binding_from_manifest(
    rationale: Mapping[str, Any],
) -> RationalePromptBinding:
    """Read canonical prompt fields, with one legacy all-missing fallback.

    Old Y6 manifests contain none of the three fields and are bound to the
    baseline prompt.  A partially populated new manifest is rejected rather
    than silently mixing a prompt identifier, body, and hash from different
    experiments.
    """

    present = [key in rationale for key in MANIFEST_PROMPT_KEYS]
    if not any(present):
        return baseline_prompt_binding()
    if not all(present):
        missing = [
            key
            for key, is_present in zip(MANIFEST_PROMPT_KEYS, present, strict=True)
            if not is_present
        ]
        raise ValueError(
            "rationale manifest의 prompt binding이 불완전합니다: " f"missing={missing}"
        )
    return validate_prompt_binding(
        rationale["rationale_prompt_id"],
        rationale["rationale_prompt_text"],
        rationale["rationale_prompt_sha256"],
        source="rationale manifest",
    )


def _first_present(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def prompt_binding_from_adapter(adapter: str | Path) -> RationalePromptBinding:
    """Read and verify prompt sidecars from a trained rationale adapter.

    Both sidecars are mandatory for a new adapter.  If neither exists, this is
    a historical adapter and baseline is the only permitted prompt.
    """

    adapter_path = Path(adapter)
    prompt_path = adapter_path / PROMPT_TEXT_FILENAME
    runtime_path = adapter_path / RUNTIME_CONFIG_FILENAME
    prompt_exists = prompt_path.is_file()
    runtime_exists = runtime_path.is_file()
    if not prompt_exists and not runtime_exists:
        return baseline_prompt_binding()
    if prompt_exists != runtime_exists:
        missing = runtime_path if prompt_exists else prompt_path
        raise ValueError(
            "rationale adapter prompt sidecar가 불완전합니다: " f"missing={missing}"
        )

    text = prompt_path.read_text(encoding="utf-8")
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{runtime_path}: JSON 파싱 실패") from exc
    if not isinstance(runtime, dict):
        raise ValueError(f"{runtime_path}: JSON object여야 합니다")

    # Canonical names are written by the current trainer.  The short aliases
    # keep already-created development sidecars readable while the canonical
    # fields written to the submission manifest remain unique.
    prompt_id = _first_present(
        runtime,
        ("rationale_prompt_id", "prompt_id", "prompt_name"),
    )
    configured_text = _first_present(
        runtime,
        ("rationale_prompt_text", "prompt_text", "prompt_template"),
    )
    sha256 = _first_present(
        runtime,
        (
            "rationale_prompt_sha256",
            "prompt_sha256",
            "prompt_template_sha256",
            "template_sha256",
        ),
    )
    if configured_text is None:
        raise ValueError(
            f"{runtime_path}: exact rationale prompt text가 config에 없습니다"
        )
    if configured_text != text:
        raise ValueError(
            f"{runtime_path}: config의 prompt text가 {prompt_path.name}과 다릅니다"
        )
    return validate_prompt_binding(
        prompt_id,
        text,
        sha256,
        source=str(runtime_path),
    )


def bind_adapter_prompt_to_manifest(
    rationale: MutableMapping[str, Any], adapter: str | Path
) -> RationalePromptBinding:
    """Bind an adapter sidecar to a staged manifest, rejecting declarations that drift."""

    adapter_binding = prompt_binding_from_adapter(adapter)
    if any(key in rationale for key in MANIFEST_PROMPT_KEYS):
        declared = prompt_binding_from_manifest(rationale)
        if declared != adapter_binding:
            raise ValueError(
                "manifest와 rationale adapter의 prompt binding이 다릅니다: "
                f"manifest={declared.prompt_id}@{declared.sha256}, "
                f"adapter={adapter_binding.prompt_id}@{adapter_binding.sha256}"
            )
    rationale.update(adapter_binding.manifest_fields())
    return adapter_binding
