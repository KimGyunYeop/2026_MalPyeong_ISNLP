"""제출 구성 manifest.

모델 asset은 이 폴더에 커밋하지 않는다. manifest가 checkpoint 경로만 가리키고, 서버와 로컬
harness가 같은 manifest를 읽어 완전히 같은 조합을 만든다. 그래야 로컬 점수와 제출 점수가
갈라지지 않는다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from main_code_submission.artifact_integrity import (
    artifact_tree_fingerprint,
    checkpoint_fingerprint,
    require_fingerprint,
)
from main_code_relonation.prompts import DEFAULT_SKELETON_HINT
from main_code_submission.rationale_prompt_binding import (
    baseline_prompt_binding,
    prompt_binding_from_adapter,
    prompt_binding_from_manifest,
    validate_prompt_binding,
)

# BF16 기준 대략적인 백본 가중치 크기(GiB). 48GB L40S 예산 판정에만 쓰는 보수적 추정이다.
BF16_GIB_PER_BILLION = 2.0
# 4-bit NF4 + 양자화 상수. 보수적으로 잡는다.
NF4_GIB_PER_BILLION = 0.7
L40S_TOTAL_GIB = 48.0

# Y6를 선발하고 기존 Docker 400편을 검증할 때 사용한 최종 출력 규칙이다.
# 운영측 사사오입은 *실수로 제출된* 영역 점수에 적용된다. 이 서버는 여기서 선택한
# 정수 3개를 출력하므로 운영측 사사오입은 항등 변환이고, 내부 정수 선택 방법은 참가자
# 모델의 자율 후처리에 해당한다. 이 값을 바꾸면 검증된 Y6 예측 400편이 달라진다.
SUBMISSION_SCORE_POSTPROCESS = "average_matched"

# 최종 Y6가 학습·검증된 입력 표면 하나만 제출 이미지에서 허용한다.
DEPLOYABLE_ESSAY_SURFACES = ("official_raw",)
# CUDA context, KV cache, 활성값, 근거 생성 여유. 실측이 아니라 사전 gate다.
RUNTIME_RESERVE_GIB = 8.0


def _checkpoint_uses_qlora(checkpoint: Path) -> bool | None:
    """Read the precision contract recorded by a score checkpoint.

    ``None`` is reserved for lightweight historical/test directories that do
    not contain ``config.json``.  A real checkpoint with a malformed config
    fails closed instead of being budgeted as BF16 by accident.
    """

    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"score checkpoint config를 읽을 수 없습니다: {config_path}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"score checkpoint config가 JSON object가 아닙니다: {config_path}"
        )
    value = raw.get("use_qlora", False)
    if not isinstance(value, bool):
        raise ValueError(
            f"score checkpoint use_qlora는 boolean이어야 합니다: "
            f"{config_path}: {value!r}"
        )
    return value


@dataclass(frozen=True)
class ScoreMember:
    """앙상블 한 구성원. `main_code`의 완주 checkpoint 폴더를 그대로 가리킨다."""

    name: str
    checkpoint: Path
    # 같은 백본을 공유하는 구성원끼리는 백본을 한 번만 로드한다. 값이 같으면 공유로 본다.
    backbone_key: str
    parameters_billion: float
    weight: float = 1.0
    # Source manifest에서 생략하면 checkpoint config의 ``use_qlora``를 읽어
    # load_manifest()가 채운다. 기존 non-QLoRA manifest는 따라서 False를 유지한다.
    # 명시값과 checkpoint가 다르면 validate()가 기동 전에 중단한다.
    load_in_4bit: bool = False

    @property
    def weights_gib(self) -> float:
        per_billion = NF4_GIB_PER_BILLION if self.load_in_4bit else BF16_GIB_PER_BILLION
        return self.parameters_billion * per_billion


@dataclass(frozen=True)
class RationaleSpec:
    """근거 생성 어댑터. 점수 백본 중 하나를 재사용하면 추가 VRAM이 거의 없다."""

    base_model: str
    adapter: Path | None
    # 근거 학습 때 사용한 immutable base revision. ``main``도 허용하지만 공유 경로는
    # 실제로 해석된 commit hash가 score CausalLM과 같은지 load 시 확인한다.
    base_model_revision: str = "main"
    # 점수 구성원과 같은 백본을 쓰면 그 `backbone_key`를 적는다. None이면 별도 로드다.
    share_backbone_key: str | None = None
    max_new_tokens: int = 512
    # 근거 어댑터는 `main_code_relonation`이 chat template 위에서 학습한다. 학습 때 쓴
    # template kwargs를 그대로 넘겨야 serving 토큰열이 학습 토큰열과 같아진다.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # 학습 artifact의 `chat_template_hash`. 값이 있으면 로드 시 실제 tokenizer의 template
    # 해시와 비교해 다르면 즉시 중단한다. 조용한 train/serve 불일치를 막는 유일한 gate다.
    chat_template_sha256: str = ""
    # 근거 prompt도 adapter가 학습한 모델 artifact다. 원문 자체와 해시를 함께 들고 다녀
    # source code가 바뀌어도 이미 학습한 adapter의 입력이 조용히 바뀌지 않게 한다.
    # 세 값이 모두 비어 있는 직접 생성/구 manifest만 historical baseline으로 해석한다.
    rationale_prompt_id: str = ""
    rationale_prompt_text: str = ""
    rationale_prompt_sha256: str = ""
    # 출력 스켈레톤 자리표시자의 길이 지시. 학습 recipe의 `rationale_skeleton_hint`와
    # 반드시 같아야 한다. 기본값은 역사적 문자열이므로 기존 manifest 동작은 불변이다.
    rationale_skeleton_hint: str = DEFAULT_SKELETON_HINT
    # prompt+completion 상한. 학습 recipe의 max_length와 같게 둔다.
    max_length: int = 8192
    # 근거 생성 전체(첫 시도 + 재시도 2회)에 허용하는 벽시계 예산(초). 0이면 무제한.
    #
    # 왜: `NEVER_DISCARD_A_SCORE.md` §6이 지목한 **남은 최대 위험**이 직렬화 락과
    # 평가 서버의 미공개 요청 timeout이다. 근거 생성이 길어지면 그 편이 우리 잘못
    # 없이 0점(제곱오차 12.16)이 된다. 예산을 넘기면 얻은 근거만 싣고 못 얻은 trait은
    # template으로 메운다. 잃는 것은 그 trait의 Judge 점수뿐이고, Judge는 종합
    # 가중치의 10%다. 확정 0점과 비교가 되지 않는다.
    deadline_seconds: float = 0.0
    parameters_billion: float = 0.0
    # 별도 근거 백본에만 적용한다. ``share_backbone_key``가 있으면 score CausalLM의 정밀도를
    # 그대로 공유하므로 True를 허용하지 않는다.
    load_in_4bit: bool = False
    # 근거를 생성하지 않는 진단 모드. 점수만 내보내면 공식 파서는 통과하지만 LLM Judge
    # 10%와 말평 아레나에서 불리하므로 실제 제출에는 쓰지 않는다.
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.deadline_seconds < 0:
            raise ValueError("rationale.deadline_seconds는 음수일 수 없습니다")
        if not (
            self.rationale_prompt_id
            or self.rationale_prompt_text
            or self.rationale_prompt_sha256
        ):
            baseline = baseline_prompt_binding()
            object.__setattr__(self, "rationale_prompt_id", baseline.prompt_id)
            object.__setattr__(self, "rationale_prompt_text", baseline.text)
            object.__setattr__(self, "rationale_prompt_sha256", baseline.sha256)


@dataclass(frozen=True)
class SubmissionConfig:
    name: str
    score_members: tuple[ScoreMember, ...]
    rationale: RationaleSpec
    essay_surface: str = "official_raw"
    served_model_name: str = "malpyeong-writing-scorer"
    # 검증된 Y6 Docker 출력과 같은 평균 정합 정수 삼중으로 고정한다.
    score_postprocess: str = SUBMISSION_SCORE_POSTPROCESS
    # --- 정수 총점 offset (2026-08-21) -----------------------------------
    # average_matched의 목표 정수 합에 더하는 **정수**. 예측 평균이 offset/3만큼
    # 이동한다. 근거는 main_code/utils.py의 average_matched_integer_scores docstring.
    #
    # 요약
    #   리더보드 (RMSE 0.5191, ρ 0.7340)을 열화 모형 6종에서 동시에 맞추면 평균 편향이
    #   0.273~0.285로 모인다. b=1/6(손익분기)을 강제하면 ρ가 0.645~0.704까지 내려가
    #   관측과 양립하지 않는다. 그래서 +1(=예측 +1/3)이 RMSE만 옮긴다.
    #
    #   **정수**여야 한다. 1/3의 배수가 아닌 이동은 essay마다 T를 다르게 움직여 동점
    #   구조를 바꾼다(실측 δ=0.15에서 ρ -0.0447). 정수 T를 같이 옮기면 400/400편의
    #   순위 벡터가 비트 단위로 같다.
    #
    #   **문항 게이트를 걸지 않는다.** 미학습 문항만 올리면 두 집단이 서로 어긋나
    #   움직여 순위 보존이 깨지고, MSE 이득 차이는 0.0025뿐이다.
    #
    # 0이면 완전히 꺼져 기존 배포와 bit-exact 동일하다.
    integer_total_offset: int = 0
    # 평가 서버가 넘기는 값. 로컬 harness가 같은 값을 쓰도록 여기에 고정한다.
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    max_tokens: int = 512
    stop: tuple[str, ...] = ("Q:", "User:")
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def unique_backbones(self) -> dict[str, float]:
        """실제 로드되는 백본 가중치 GiB.

        `engine.load()`는 같은 `backbone_key`를 가진 구성원을
        `load_shared_backbone_checkpoints`로 **backbone 한 벌**에 올리고 LoRA 어댑터만 갈아
        끼운다. 따라서 같은 키는 한 번만 센다. score head는 보통 수 MB라 무시한다.

        동등성은 `main_code/tests/test_shared_backbone_ensemble.py`가 보장한다. 공유 경로 점수가
        독립 로드 점수와 1e-6 안에서 같음을 확인하며, `activate`를 빼먹으면 점수가 달라진다는
        것도 함께 고정한다.
        """

        sizes: dict[str, float] = {}
        for member in self.score_members:
            # 같은 백본을 공유하는 구성원 중 가장 큰 값을 쓴다. 정상 구성에서는 모두 같다.
            key = f"backbone::{member.backbone_key}"
            sizes[key] = max(sizes.get(key, 0.0), member.weights_gib)
        if self.rationale.enabled and self.rationale.share_backbone_key is None:
            per_billion = (
                NF4_GIB_PER_BILLION
                if self.rationale.load_in_4bit
                else BF16_GIB_PER_BILLION
            )
            sizes[f"rationale::{self.rationale.base_model}"] = (
                self.rationale.parameters_billion * per_billion
            )
        return sizes

    def estimated_weights_gib(self) -> float:
        return sum(self.unique_backbones().values())

    def vram_budget_report(self) -> dict[str, Any]:
        """48GB 예산 사전 gate. 실측 peak가 아니라 로드 전 판정이다."""

        weights = self.estimated_weights_gib()
        budget = L40S_TOTAL_GIB - RUNTIME_RESERVE_GIB
        return {
            "unique_backbones": self.unique_backbones(),
            "score_load_in_4bit": {
                member.name: member.load_in_4bit for member in self.score_members
            },
            "estimated_weights_gib": round(weights, 2),
            "budget_gib": round(budget, 2),
            "fits": weights <= budget,
            "headroom_gib": round(budget - weights, 2),
        }

    def validate(self) -> "SubmissionConfig":
        # 설정 오류를 파일시스템 오류보다 먼저 본다. checkpoint가 없는 것은 환경 문제이고
        # 표면이 틀린 것은 계약 위반이라 진단 우선순위가 다르다.
        if self.score_postprocess != SUBMISSION_SCORE_POSTPROCESS:
            raise ValueError(
                "제출 score_postprocess는 average_matched로 고정됩니다: "
                f"{self.score_postprocess!r}"
            )
        if self.essay_surface not in DEPLOYABLE_ESSAY_SURFACES:
            # 원천 문단 배열이 필요한 surface(canonical)는 제출 시 만들 수 없다. 학습/서빙
            # 표면이 갈리면 organization RMSE가 `.54 -> 1.03`으로 무너진 전례가 있다.
            raise ValueError(
                f"제출 essay_surface는 {DEPLOYABLE_ESSAY_SURFACES} 중 하나여야 합니다: "
                f"{self.essay_surface!r}"
            )
        if not self.score_members:
            raise ValueError("score_members가 비었습니다")
        configured_prompt = validate_prompt_binding(
            self.rationale.rationale_prompt_id,
            self.rationale.rationale_prompt_text,
            self.rationale.rationale_prompt_sha256,
            source="submission rationale config",
        )
        names = [member.name for member in self.score_members]
        if len(set(names)) != len(names):
            raise ValueError(f"score_members 이름이 중복됩니다: {names}")
        precision_by_backbone: dict[str, bool] = {}
        for member in self.score_members:
            if not member.checkpoint.is_dir():
                raise FileNotFoundError(
                    f"checkpoint 폴더가 없습니다: {member.checkpoint}"
                )
            if not isinstance(member.load_in_4bit, bool):
                raise ValueError(f"{member.name}: load_in_4bit는 boolean이어야 합니다")
            if not math.isfinite(member.parameters_billion) or (
                member.parameters_billion <= 0
            ):
                raise ValueError(
                    f"{member.name}: parameters_billion은 양의 유한값이어야 합니다"
                )
            if member.weight <= 0:
                raise ValueError(f"{member.name}: weight는 양수여야 합니다")
            checkpoint_precision = _checkpoint_uses_qlora(member.checkpoint)
            if checkpoint_precision is None:
                if member.load_in_4bit:
                    raise FileNotFoundError(
                        f"{member.name}: 4-bit score checkpoint config.json이 없습니다: "
                        f"{member.checkpoint}"
                    )
            elif checkpoint_precision != member.load_in_4bit:
                raise ValueError(
                    f"{member.name}: manifest load_in_4bit={member.load_in_4bit!r} != "
                    f"checkpoint use_qlora={checkpoint_precision!r}"
                )
            previous_precision = precision_by_backbone.setdefault(
                member.backbone_key, member.load_in_4bit
            )
            if previous_precision != member.load_in_4bit:
                raise ValueError(
                    f"{member.backbone_key}: 공유 score backbone 구성원의 "
                    "load_in_4bit가 서로 다릅니다"
                )
        if self.rationale.enabled:
            if self.rationale.adapter is None:
                raise ValueError(
                    "근거 생성이 enabled이면 학습된 rationale adapter가 필요합니다"
                )
            if self.rationale.share_backbone_key is not None:
                keys = {member.backbone_key for member in self.score_members}
                if self.rationale.share_backbone_key not in keys:
                    raise ValueError(
                        "rationale.share_backbone_key가 score_members의 backbone_key에 "
                        f"없습니다: {self.rationale.share_backbone_key} not in {sorted(keys)}"
                    )
                if self.rationale.load_in_4bit:
                    raise ValueError(
                        "공유 rationale는 score CausalLM과 같은 정밀도를 사용합니다. "
                        "load_in_4bit를 false로 두십시오"
                    )
            if (
                self.rationale.adapter is not None
                and not self.rationale.adapter.is_dir()
            ):
                raise FileNotFoundError(
                    f"근거 어댑터가 없습니다: {self.rationale.adapter}"
                )
            if self.rationale.adapter is not None:
                adapter_prompt = prompt_binding_from_adapter(self.rationale.adapter)
                if adapter_prompt != configured_prompt:
                    raise ValueError(
                        "submission manifest와 rationale adapter의 prompt binding이 "
                        "다릅니다: "
                        f"manifest={configured_prompt.prompt_id}@"
                        f"{configured_prompt.sha256}, adapter={adapter_prompt.prompt_id}@"
                        f"{adapter_prompt.sha256}"
                    )
        if self.max_tokens < 1 or self.max_tokens > 2048:
            raise ValueError("max_tokens는 1~2048이어야 합니다")
        budget = self.vram_budget_report()
        if not budget["fits"]:
            raise ValueError(f"L40S static VRAM budget을 초과합니다: {budget}")
        _validate_deployed_artifacts(self)
        return self


def _validate_deployed_artifacts(config: SubmissionConfig) -> None:
    """Docker staging manifest가 선언한 실제 배포 asset을 load 전에 재검증한다."""

    required = config.extra.get("artifact_integrity_required", False)
    if not required:
        return
    if required is not True:
        raise ValueError("artifact_integrity_required는 true여야 합니다")
    if not config.rationale.enabled or config.rationale.adapter is None:
        raise ValueError("배포 manifest는 학습된 rationale adapter를 활성화해야 합니다")

    member_names = {member.name for member in config.score_members}
    expected_core = config.extra.get("deployed_checkpoint_artifacts")
    expected_closures = config.extra.get("deployed_checkpoint_closures")
    if not isinstance(expected_core, dict) or set(expected_core) != member_names:
        raise ValueError(
            "deployed_checkpoint_artifacts 이름이 score_members와 다릅니다: "
            f"expected={sorted(member_names)}, "
            f"actual={sorted(expected_core) if isinstance(expected_core, dict) else None}"
        )
    if (
        not isinstance(expected_closures, dict)
        or set(expected_closures) != member_names
    ):
        raise ValueError(
            "deployed_checkpoint_closures 이름이 score_members와 다릅니다: "
            f"expected={sorted(member_names)}, "
            "actual="
            f"{sorted(expected_closures) if isinstance(expected_closures, dict) else None}"
        )
    for member in config.score_members:
        require_fingerprint(
            label=f"score checkpoint {member.name}",
            expected=expected_core[member.name],
            actual=checkpoint_fingerprint(member.checkpoint),
        )
        require_fingerprint(
            label=f"score checkpoint closure {member.name}",
            expected=expected_closures[member.name],
            actual=artifact_tree_fingerprint(member.checkpoint),
        )

    if config.rationale.adapter is not None:
        require_fingerprint(
            label="rationale adapter",
            expected=config.extra.get("deployed_rationale_artifact"),
            actual=artifact_tree_fingerprint(config.rationale.adapter),
        )


def load_manifest(path: str | Path) -> SubmissionConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    root = Path(raw.get("root", ".")).expanduser()
    extra_raw = raw.get("extra", {})
    if not isinstance(extra_raw, dict):
        raise ValueError("submission manifest extra는 object여야 합니다")
    # Dockerfile이 고정하는 배포 root에서는 integrity block 누락/false를 허용하지 않는다.
    # 연구용 host manifest는 build 전 source 경로를 읽어야 하므로 이 gate의 대상이 아니다.
    if root == Path("/opt/submission") and (
        extra_raw.get("artifact_integrity_required") is not True
    ):
        raise ValueError(
            "/opt/submission manifest에는 artifact_integrity_required=true가 필요합니다"
        )

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (root / candidate)

    def parse_score_member(item: dict[str, Any]) -> ScoreMember:
        checkpoint = resolve(item["checkpoint"])
        checkpoint_precision = _checkpoint_uses_qlora(checkpoint)
        declared_precision = item.get("load_in_4bit")
        if declared_precision is not None and not isinstance(declared_precision, bool):
            raise ValueError(
                f"{item.get('name', '<unnamed>')}: load_in_4bit는 boolean이어야 합니다"
            )
        if checkpoint_precision is not None:
            if (
                declared_precision is not None
                and declared_precision != checkpoint_precision
            ):
                raise ValueError(
                    f"{item.get('name', '<unnamed>')}: manifest load_in_4bit="
                    f"{declared_precision!r} != checkpoint use_qlora="
                    f"{checkpoint_precision!r}"
                )
            load_in_4bit = checkpoint_precision
        else:
            load_in_4bit = (
                bool(declared_precision) if declared_precision is not None else False
            )
        return ScoreMember(
            name=item["name"],
            checkpoint=checkpoint,
            backbone_key=item.get("backbone_key", item["name"]),
            parameters_billion=float(item.get("parameters_billion", 0.0)),
            weight=float(item.get("weight", 1.0)),
            load_in_4bit=load_in_4bit,
        )

    members = tuple(parse_score_member(item) for item in raw["score_members"])
    rationale_raw = raw.get("rationale", {})
    if not isinstance(rationale_raw, dict):
        raise ValueError("submission manifest rationale는 object여야 합니다")
    rationale_prompt = prompt_binding_from_manifest(rationale_raw)
    rationale = RationaleSpec(
        base_model=rationale_raw.get("base_model", ""),
        adapter=(
            resolve(rationale_raw["adapter"]) if rationale_raw.get("adapter") else None
        ),
        base_model_revision=str(rationale_raw.get("base_model_revision", "main")),
        share_backbone_key=rationale_raw.get("share_backbone_key"),
        load_in_4bit=bool(rationale_raw.get("load_in_4bit", False)),
        max_new_tokens=int(rationale_raw.get("max_new_tokens", 512)),
        deadline_seconds=float(rationale_raw.get("deadline_seconds", 0.0)),
        chat_template_kwargs=dict(rationale_raw.get("chat_template_kwargs") or {}),
        chat_template_sha256=str(rationale_raw.get("chat_template_sha256", "")),
        rationale_prompt_id=rationale_prompt.prompt_id,
        rationale_prompt_text=rationale_prompt.text,
        rationale_prompt_sha256=rationale_prompt.sha256,
        rationale_skeleton_hint=str(
            rationale_raw.get("rationale_skeleton_hint") or DEFAULT_SKELETON_HINT
        ),
        max_length=int(rationale_raw.get("max_length", 8192)),
        parameters_billion=float(rationale_raw.get("parameters_billion", 0.0)),
        enabled=bool(rationale_raw.get("enabled", True)),
    )
    return SubmissionConfig(
        name=raw.get("name", "submission"),
        score_members=members,
        rationale=rationale,
        essay_surface=raw.get("essay_surface", "official_raw"),
        served_model_name=raw.get("served_model_name", "malpyeong-writing-scorer"),
        # 누락도 조용히 default로 보정하지 않는다. 최종 manifest가 계약을 직접 선언해야 한다.
        score_postprocess=str(raw.get("score_postprocess", "")),
        integer_total_offset=int(raw.get("integer_total_offset", 0)),
        max_tokens=int(raw.get("max_tokens", 512)),
        seed=int(raw.get("seed", 42)),
        notes=raw.get("notes", ""),
        extra=extra_raw,
    ).validate()
