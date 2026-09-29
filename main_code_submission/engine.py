"""점수 앙상블 + 어댑터 스왑 근거 생성 오케스트레이션.

점수 계산은 `main_code`의 검증된 경로를 그대로 호출한다(현행 제출 계약의 명시적 공유). 이 파일이
새로 하는 일은 세 가지다.

1. 공식 단일 user message에서 꺼낸 prompt/essay를 `main_code`가 학습 때 본 것과 같은 row로
   재구성한다.
2. 여러 score checkpoint를 가중 평균한다. 같은 백본을 공유하는 구성원은 LoRA 어댑터만
   스왑해 VRAM을 한 번만 쓴다.
3. Y6 후처리로 확정한 정수 점수를 조건으로 근거를 생성하고 JSON을 만든다.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from main_code.datasets import (
    EssayRegressionDataset,
    RegressionCollator,
)
from main_code.models import (
    load_checkpoint,
    load_shared_backbone_checkpoints,
    verify_loaded_peft_adapter_exact,
)

from .config import ScoreMember, SubmissionConfig, SUBMISSION_SCORE_POSTPROCESS
from .degrade import (
    NEUTRAL_SUBMITTED_SCORES,
    fill_missing_rationales,
    last_resort_outputs,
)
from main_code_relonation.prompts import DEFAULT_SKELETON_HINT

from .prompts import build_rationale_messages
from .request_parser import OfficialRequestText
from main_code.postprocess import ScorePostprocessor
from .schema import (
    FALLBACK_RATIONALE,
    TRAITS,
    TraitOutput,
    enforce_official_parse,
)

LOGGER = logging.getLogger(__name__)

# 역사 Y6와 같은 정수 출력 규칙을 한 곳에서 공유한다.
_POSTPROCESSOR = ScorePostprocessor(SUBMISSION_SCORE_POSTPROCESS)
# 일부 CUDA/torch 조합에서는 greedy 생성의 첫 토큰이 EOS와 수치적으로 경합해, 같은
# checkpoint·prompt인데도 보이는 completion이 한 글자도 나오지 않는 경우가 있었다. 재시도에서
# EOS를 JSON prefix가 자리 잡을 만큼만 막는다. 정상 생성에는 이 값이 전혀 적용되지 않는다.
_EMPTY_COMPLETION_RETRY_MIN_NEW_TOKENS = 8
# greedy 재시도는 completion이 **비었을 때만** 다른 결과를 낸다. 비어 있지 않은데 JSON이
# 깨진 경우(escape 안 된 인용부호, max_new_tokens 절단)에는 같은 토큰이 그대로 다시 나오므로
# 마지막 한 번은 표본추출로 실제로 다른 경로를 밟는다. 전역 RNG를 이 시드로 고정해 앞선
# 요청 이력과 무관하게 결정적이다.
_SAMPLED_RETRY_SEED = 20260816
_SAMPLED_RETRY_TEMPERATURE = 0.7
_SAMPLED_RETRY_TOP_P = 0.9
# 예산 초과 essay를 잘라 넣을 때의 축소 계수와 시도 횟수. 1,200자 계약에서는 실제로 도달할
# 일이 없지만(관측 최대 사용률 25.2%), 도달했을 때 근거를 포기하는 대신 앞부분만 보고
# 생성하게 한다. 점수는 이 경로와 무관하게 이미 확정되어 있다.
_BUDGET_TRUNCATION_MARGIN = 0.95
_BUDGET_TRUNCATION_ATTEMPTS = 6
# 직렬화 큐 대기가 이 값을 넘으면 경고한다. 로컬 실측 2.488초/편의 4배로, 동시 요청이
# 실제로 쌓이고 있을 때만 걸리도록 잡았다.
_LOCK_WAIT_WARN_SECONDS = 10.0
# 학습 스켈레톤의 자리표시자를 그대로 뱉은 trait을 골라내는 패턴.
#
# 예전에는 `<{trait} 근거, 두 문장 이내 180자 이내>` 리터럴 하나만 봤다. 길이 지시가
# prompt 판본별로 달라질 수 있게 되었으므로(`rationale_skeleton_hint`) 리터럴 대신
# 자리표시자의 **구조**를 본다. 기본 hint의 리터럴도 이 패턴에 그대로 걸리므로 기존
# 판정은 보존되고, 새 hint를 쓰는 어댑터까지 함께 검출된다. 이 함수들은 module-level이라
# config를 볼 수 없고, 놓치는 것보다 넓게 잡는 쪽이 안전하다.
_RATIONALE_TEMPLATE_PLACEHOLDER_RE = {
    trait: re.compile(rf"<{re.escape(trait)}\s*근거,[^>]*>") for trait in TRAITS
}


def _contains_template_placeholder(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in _RATIONALE_TEMPLATE_PLACEHOLDER_RE.values()
    )


def official_row(text: OfficialRequestText) -> dict[str, Any]:
    """평가 요청을 `main_code`가 아는 schema 없는 inference row로 바꾼다.

    `main_code.datasets`는 schema 없는 공식 row의 `essay`를 raw string으로 그대로 통과시킨다.
    prompt/essay의 어떤 문자도 정규화하지 않는다.
    """

    return {"id": "request", "prompt": text.prompt, "essay": text.essay}


@dataclass
class _LoadedMember:
    member: ScoreMember
    scorer: Any  # main_code.models.RegressionScorer
    tokenizer: Any
    config: Any  # main_code.config.RegressionConfig
    collator: Any  # main_code.datasets.RegressionCollator
    # 공유 backbone일 때만 채운다. 채점 직전에 이 어댑터를 활성화해야 한다.
    ensemble: Any | None = None
    adapter_name: str | None = None

    def activate(self) -> None:
        """공유 backbone이면 이 구성원의 어댑터를 활성화한다.

        빼먹으면 **다른 구성원의 점수가 조용히 나온다.** `main_code/tests/
        test_shared_backbone_ensemble.py`가 그 사실을 test로 고정한다.
        """

        if self.ensemble is not None and self.adapter_name is not None:
            self.ensemble.activate(self.adapter_name)


class SubmissionEngine:
    """서버와 로컬 harness가 공유하는 단일 추론 엔진."""

    def __init__(self, config: SubmissionConfig, *, device: str | None = None) -> None:
        self.config = config.validate()
        # 정수 총점 offset은 postprocessor 안에서 적용된다. 0이면 기존과 bit-exact.
        # 별도 경로를 두지 않는 것이 중요하다 — offset을 더하는 곳이 두 군데면
        # 한쪽만 켜졌는지 확인할 방법이 없다.
        self._score_postprocessor = ScorePostprocessor(
            self.config.score_postprocess,
            integer_total_offset=int(
                getattr(self.config, "integer_total_offset", 0) or 0
            ),
        )
        if device is None:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                raise RuntimeError(
                    "제출 SubmissionEngine은 CUDA GPU가 필요합니다. "
                    "CPU fallback은 허용하지 않습니다."
                )
            device = "cuda"
        # 명시적 device="cpu"는 모델 없는 단위 테스트와 진단 코드에만 남긴다.
        # 실제 server는 device를 넘기지 않으므로 위 CUDA-required 계약을 반드시 탄다.
        self.device = torch.device(device)
        self._members: list[_LoadedMember] = []
        self._ensembles: list[Any] = []
        self._ensembles_by_backbone: dict[str, Any] = {}
        self._rationale_model: Any | None = None
        self._rationale_tokenizer: Any | None = None
        self._rationale_ensemble: Any | None = None
        self._rationale_adapter_name: str | None = None
        self._loaded = False
        # 규정 §5는 "여러 추론 요청을 순차 또는 소수 병렬로 처리"를 요구한다. 순차가 허용되므로
        # 한 번에 하나만 통과시켜 동시 요청이 48GB에서 KV cache를 겹쳐 쌓지 못하게 한다.
        self._inference_lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        """모든 점수 구성원과 근거 어댑터를 한 번에 로드한다.

        로드 오류는 그대로 호출자에게 전파한다. 제출 서버는 ``load()``가 끝난 뒤에만
        HTTP 서버를 시작하므로, 부분 앙상블이나 학습되지 않은 근거 모델이 서빙될
        경로는 없다.
        """

        if self._loaded:
            return
        # 같은 backbone_key를 쓰는 구성원은 backbone을 한 번만 올리고 LoRA 어댑터만 갈아 끼운다.
        # 7B 기준 구성원당 약 14.5GiB를 절약하므로 48GB 안에서 3종 이상 앙상블이 가능해진다.
        groups: dict[str, list[ScoreMember]] = {}
        for member in self.config.score_members:
            groups.setdefault(member.backbone_key, []).append(member)
        for backbone_key, group in groups.items():
            rationale_share = (
                self.config.rationale.enabled
                and self.config.rationale.share_backbone_key == backbone_key
            )
            if len(group) > 1 or rationale_share:
                self._members.extend(
                    self._load_shared_group(
                        group,
                        retain_causal_lm=rationale_share,
                    )
                )
            else:
                self._members.append(self._load_member(group[0]))
        if len(self._members) != len(self.config.score_members):
            raise RuntimeError(
                "score member 로드 수가 manifest와 다릅니다: "
                f"loaded={len(self._members)}, expected={len(self.config.score_members)}"
            )
        if self.config.rationale.enabled:
            self._load_rationale()
            if self._rationale_model is None or self._rationale_tokenizer is None:
                raise RuntimeError("근거 모델/tokenizer가 완전히 로드되지 않았습니다")
        self._loaded = True

    def _check_surface(self, name: str, config: Any) -> None:
        # 연구 loader의 encoder/encoder-decoder 호환은 보존하되 제출은 생성과 동일한
        # decoder-only CausalLM 계약으로 닫는다. 지원하지 않는 artifact를 조용히 다른
        # architecture로 올리는 것보다 이미지 기동을 실패시키는 편이 안전하다.
        if config.backbone_type != "decoder":
            raise ValueError(
                f"{name}: 제출 score checkpoint는 decoder만 지원합니다: "
                f"{config.backbone_type!r}"
            )
        if config.essay_surface != self.config.essay_surface:
            raise ValueError(
                f"{name}: checkpoint surface {config.essay_surface!r} != 제출 surface "
                f"{self.config.essay_surface!r}. 배포 동등성이 깨집니다."
            )

    @staticmethod
    def _check_score_precision(member: ScoreMember, config: Any) -> None:
        """Keep manifest budgeting and the checkpoint runtime on one dtype path."""

        checkpoint_uses_qlora = getattr(config, "use_qlora", None)
        if not isinstance(checkpoint_uses_qlora, bool):
            raise ValueError(
                f"{member.name}: checkpoint use_qlora가 boolean이 아닙니다: "
                f"{checkpoint_uses_qlora!r}"
            )
        if checkpoint_uses_qlora != member.load_in_4bit:
            raise ValueError(
                f"{member.name}: manifest load_in_4bit={member.load_in_4bit!r} != "
                f"checkpoint use_qlora={checkpoint_uses_qlora!r}"
            )

    def _load_member(self, member: ScoreMember) -> "_LoadedMember":
        loaded = load_checkpoint(
            member.checkpoint,
            device=self.device,
            load_in_4bit=member.load_in_4bit,
        )
        self._check_score_precision(member, loaded.config)
        self._check_surface(member.name, loaded.config)
        return _LoadedMember(
            member=member,
            scorer=loaded.scorer,
            tokenizer=loaded.tokenizer,
            config=loaded.config,
            collator=RegressionCollator(
                loaded.tokenizer,
                loaded.config,
                include_labels=False,
                max_length=loaded.config.max_length,
            ),
        )

    def _load_shared_group(
        self,
        group: list[ScoreMember],
        *,
        retain_causal_lm: bool = False,
    ) -> list["_LoadedMember"]:
        """같은 backbone을 쓰는 구성원들을 backbone 한 벌로 올린다."""

        precisions = {member.load_in_4bit for member in group}
        if len(precisions) != 1:
            raise ValueError("공유 score backbone 구성원의 load_in_4bit가 다릅니다")
        ensemble = load_shared_backbone_checkpoints(
            [member.checkpoint for member in group],
            device=self.device,
            load_in_4bit=group[0].load_in_4bit,
            names=[member.name for member in group],
            with_lm_head=retain_causal_lm,
        )
        entries: list[_LoadedMember] = []
        for member, shared in zip(group, ensemble.members, strict=True):
            self._check_score_precision(member, shared.config)
            self._check_surface(member.name, shared.config)
            entries.append(
                _LoadedMember(
                    member=member,
                    scorer=shared.scorer,
                    tokenizer=shared.tokenizer,
                    config=shared.config,
                    collator=RegressionCollator(
                        shared.tokenizer,
                        shared.config,
                        include_labels=False,
                        max_length=shared.config.max_length,
                    ),
                    ensemble=ensemble,
                    adapter_name=shared.adapter_name,
                )
            )
        self._ensembles.append(ensemble)
        self._ensembles_by_backbone[group[0].backbone_key] = ensemble
        LOGGER.info(
            "공유 backbone %s: 구성원 %d개를 backbone 한 벌로 로드",
            group[0].backbone_key,
            len(group),
        )
        return entries

    def _load_rationale(self) -> None:
        spec = self.config.rationale
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        share = spec.share_backbone_key
        if share is not None:
            base = next(m for m in self._members if m.member.backbone_key == share)
            model_id = spec.base_model or base.config.model_id
            if model_id != base.config.model_id:
                raise ValueError(
                    "rationale base model이 공유 score model과 다릅니다: "
                    f"{model_id!r} != {base.config.model_id!r}"
                )
            ensemble = self._ensembles_by_backbone.get(share)
            if ensemble is None or ensemble.causal_lm is None:
                raise RuntimeError(
                    f"{share}: score backbone이 AutoModelForCausalLM으로 로드되지 않았습니다"
                )
            requested = AutoConfig.from_pretrained(
                model_id,
                revision=spec.base_model_revision,
                trust_remote_code=base.config.trust_remote_code,
            )
            requested_commit = getattr(requested, "_commit_hash", None)
            loaded_commit = getattr(ensemble.causal_lm.config, "_commit_hash", None)
            if requested_commit and loaded_commit and requested_commit != loaded_commit:
                raise ValueError(
                    "score/rationale base revision이 다릅니다: "
                    f"score={loaded_commit}, rationale={requested_commit}"
                )
            if spec.adapter is None:  # config.validate()도 막지만 직접 호출도 닫는다.
                raise ValueError("공유 rationale adapter가 필요합니다")
            tokenizer_source = str(spec.adapter)
            self._rationale_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=base.config.trust_remote_code,
            )
            self._verify_chat_template(self._rationale_tokenizer, spec)
            self._rationale_adapter_name = ensemble.load_causal_adapter(
                spec.adapter,
                adapter_name="rationale",
            )
            self._rationale_ensemble = ensemble
            self._rationale_model = ensemble.causal_lm.eval()
            LOGGER.info(
                "score/rationale가 CausalLM 한 벌을 공유합니다: %s @ %s",
                model_id,
                loaded_commit or base.config.model_revision,
            )
            return
        else:
            model_id = spec.base_model
        if not model_id:
            raise ValueError("rationale.base_model이 필요합니다")
        tokenizer_source = (
            str(spec.adapter)
            if spec.adapter is not None
            and (spec.adapter / "tokenizer_config.json").is_file()
            else model_id
        )
        self._rationale_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=(
                None if tokenizer_source != model_id else spec.base_model_revision
            ),
            trust_remote_code=True,
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "revision": spec.base_model_revision,
        }
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = {"": self.device.index or 0}
        self._verify_chat_template(self._rationale_tokenizer, spec)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if spec.adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(spec.adapter))
            verify_loaded_peft_adapter_exact(model, spec.adapter)
        if not spec.load_in_4bit:
            model = model.to(self.device)
        self._rationale_model = model.eval()

    def _verify_chat_template(self, tokenizer: Any, spec: Any) -> None:
        """학습 artifact의 chat template 해시와 서빙 tokenizer의 해시를 비교한다.

        어댑터는 template이 만든 토큰열 위에서 학습됐다. tokenizer revision이 바뀌어 template
        문자열이 달라지면 같은 message가 다른 토큰열이 되고 어댑터 품질이 조용히 무너진다.
        `main_code_relonation.data`가 학습 때 같은 방식으로 해시를 기록한다.
        """

        template = getattr(tokenizer, "chat_template", None)
        if not template:
            raise ValueError(
                f"{spec.base_model}: tokenizer에 chat_template이 없습니다. 근거 어댑터는 "
                "chat template 위에서 학습되므로 template 없는 backbone은 쓸 수 없습니다."
            )
        digest = hashlib.sha256(str(template).encode("utf-8")).hexdigest()
        if spec.chat_template_sha256 and digest != spec.chat_template_sha256:
            raise ValueError(
                "chat template이 학습 때와 다릅니다: "
                f"expected={spec.chat_template_sha256} got={digest}. "
                "근거 어댑터를 학습한 tokenizer revision을 그대로 쓰십시오."
            )
        self._rationale_template_sha256 = digest

    # --- scoring ---------------------------------------------------------
    @torch.no_grad()
    def score(self, text: OfficialRequestText) -> dict[str, float]:
        """가중 평균한 소수 점수. 반올림하지 않는다."""

        if not self._members:
            raise RuntimeError("engine.load()를 먼저 호출해야 합니다")
        row = official_row(text)
        totals = {trait: 0.0 for trait in TRAITS}
        weight_sum = 0.0
        for entry in self._members:
            per_member = self._score_one(entry, row)
            weight = entry.member.weight
            for trait in TRAITS:
                totals[trait] += per_member[trait] * weight
            weight_sum += weight
        if weight_sum <= 0:
            raise ValueError("weight 합이 0입니다")
        return {trait: totals[trait] / weight_sum for trait in TRAITS}

    @torch.no_grad()
    def _score_one(self, entry: _LoadedMember, row: dict[str, Any]) -> dict[str, float]:
        """`infer.py`와 완전히 같은 dataset/collator/forward 경로로 한 편을 채점한다.

        손으로 tokenize하지 않는다. `essay_mask`, `sentence_ids`, `shared_pooling_mask`,
        `prompt_ids` 같은 batch key는 config에 따라 collator가 만들고 scorer가 그것을
        요구한다. 여기서 직접 만들면 pooling/organization 경로가 조용히 달라져 offline
        점수와 serving 점수가 갈라진다.
        """

        # 공유 backbone에서는 이 호출이 없으면 다른 구성원의 어댑터로 채점된다.
        entry.activate()
        config = entry.config
        dataset = EssayRegressionDataset(
            [row], config, split="inference", require_labels=False
        )
        batch = entry.collator([dataset[0]])
        model_inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key != "labels"
        }
        detail_active = config.detail_head_mode != "none"
        result = entry.scorer(
            **model_inputs,
            return_probabilities=(config.score_head == "distribution"),
            return_detail_predictions=detail_active,
        )
        values = result["scores"].detach().float().reshape(-1).tolist()
        if len(values) != len(TRAITS):
            raise ValueError(
                f"{entry.member.name}: score 개수가 {len(values)}로 3이 아닙니다"
            )
        return dict(zip(TRAITS, values, strict=True))

    # --- rationale -------------------------------------------------------
    @contextlib.contextmanager
    def _rationale_adapter_scope(self) -> Iterator[None]:
        """Activate rationale LoRA and prove the prior score adapter is restored.

        Shared score/rationale serving uses heterogeneous PEFT adapters on one
        CausalLM.  ``SharedBackboneEnsemble.using`` restores in ``finally``;
        these state assertions make a missed activation or failed restoration
        a visible rationale error instead of contaminating the next score.
        """

        ensemble = self._rationale_ensemble
        adapter_name = self._rationale_adapter_name
        if ensemble is None or adapter_name is None:
            yield
            return
        previous = ensemble.active_adapter
        if previous is None or previous not in ensemble.adapter_names:
            raise RuntimeError(
                "근거 생성 전에 활성 score adapter가 없습니다: "
                f"active={previous!r}, score_adapters={ensemble.adapter_names}"
            )
        try:
            with ensemble.using(adapter_name):
                if ensemble.active_adapter != adapter_name:
                    raise RuntimeError(
                        "근거 adapter 활성화가 반영되지 않았습니다: "
                        f"expected={adapter_name!r}, active={ensemble.active_adapter!r}"
                    )
                yield
        finally:
            # 복원 실패는 **다음** 요청의 점수를 오염시킨다. 예외를 올리는 것만으로는
            # 서버가 계속 살아 있으므로 오염이 멈추지 않는다. 그래서 먼저 강제로 되돌리고,
            # 그래도 안 되면 그때 예외를 올린다. `finally`에서 raise하면 진행 중인 예외를
            # 덮으므로 강제 복원이 성공한 경우에는 아무것도 던지지 않는다.
            if ensemble.active_adapter != previous:
                LOGGER.error(
                    "근거 생성 뒤 score adapter가 %r로 남아 강제 복원합니다 (기대 %r)",
                    ensemble.active_adapter,
                    previous,
                )
                try:
                    ensemble.activate(previous)
                except Exception:  # noqa: BLE001 - 아래 상태 검사로 판정한다
                    LOGGER.exception("score adapter 강제 복원 호출이 실패했습니다")
                if ensemble.active_adapter != previous:
                    raise RuntimeError(
                        "근거 생성 뒤 score adapter 복원에 실패했습니다: "
                        f"expected={previous!r}, active={ensemble.active_adapter!r}"
                    )

    @torch.no_grad()
    def rationales_with_raw(
        self, text: OfficialRequestText, scores: dict[str, float]
    ) -> tuple[dict[str, str], str]:
        """근거와 **원문 완성문**을 함께 돌려준다.

        원문이 필요한 이유: 구조 회수까지 전부 실패해도 모델이 쓴 문장 자체는 남아
        있다. Judge는 "generic한 총평, 상투적 표현, 템플릿형 설명은 낮게 평가하라"고
        명시하므로, template으로 덮는 것보다 원문을 정제해 쓰는 편이 낫다.
        """

        return self._rationales_impl(text, scores)

    def rationales(
        self, text: OfficialRequestText, scores: dict[str, float]
    ) -> dict[str, str]:
        """확정 점수를 조건으로 세 근거를 만든다. 점수는 절대 바꾸지 않는다."""

        return self._rationales_impl(text, scores)[0]

    def _rationales_impl(
        self, text: OfficialRequestText, scores: dict[str, float]
    ) -> tuple[dict[str, str], str]:
        if not self.config.rationale.enabled:
            return {trait: "" for trait in TRAITS}, ""
        if self._rationale_model is None or self._rationale_tokenizer is None:
            raise RuntimeError("근거 모델이 로드되지 않았습니다")
        spec = self.config.rationale
        tokenizer = self._rationale_tokenizer
        prompt_ids, truncated_chars = self._rationale_prompt_ids(text, scores)
        if truncated_chars:
            LOGGER.warning(
                "근거 prompt가 예산을 넘어 essay 뒤쪽 %d자를 잘라 생성합니다. "
                "점수는 잘리지 않은 원문으로 이미 확정되어 있습니다.",
                truncated_chars,
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": spec.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }

        # 벽시계 예산. 0이면 예전과 bit-exact 같은 무제한 경로다.
        # 필드가 없는 옛 spec/진단 fake는 무제한으로 해석한다(위 skeleton hint와 같은 규약).
        deadline_seconds = float(getattr(spec, "deadline_seconds", 0.0) or 0.0)
        deadline = (
            time.monotonic() + deadline_seconds if deadline_seconds > 0 else None
        )
        self._deadline_hits = 0

        def budget_left() -> bool:
            return deadline is None or time.monotonic() < deadline

        if deadline is not None:
            # generate() 안에서도 멈춰야 한 번의 긴 생성이 예산을 넘기지 않는다.
            # StoppingCriteria는 토큰마다 호출되므로 중단 지점이 토큰 경계다.
            from transformers import StoppingCriteria, StoppingCriteriaList

            engine = self

            class _Deadline(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    if time.monotonic() >= deadline:
                        engine._deadline_hits += 1
                        return True
                    return False

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([_Deadline()])

        def run(**overrides: Any) -> tuple[dict[str, str], str]:
            generated = self._rationale_model.generate(
                **{**generation_kwargs, **overrides}
            )
            completion = tokenizer.decode(
                generated[0][len(prompt_ids) :], skip_special_tokens=True
            )
            return parse_rationale_completion(completion), completion

        with self._rationale_adapter_scope():
            best, completion = run()
            longest = completion
            if len(best) == len(TRAITS):
                return best, longest
            # --- 1차 복구: EOS를 잠깐 막아 빈 completion만 되살린다 ----------------
            # 관측된 실패는 greedy 첫 토큰이 EOS와 경합해 completion이 0자로 나오는
            # 경우다. 정상 호출의 인자·토큰·응답은 위 경로 그대로다.
            retry_min_tokens = min(
                _EMPTY_COMPLETION_RETRY_MIN_NEW_TOKENS, spec.max_new_tokens
            )
            if not budget_left():
                # 재시도는 근거를 더 얻으려는 시도일 뿐이다. 예산을 넘겨 가며 하면
                # 그 편 전체가 timeout으로 0점이 될 수 있고, 그건 근거 세 개를 다
                # 잃는 것보다 30배 비싸다.
                LOGGER.warning(
                    "근거 예산 %.1f초를 소진해 재시도를 생략합니다 (%d/%d trait 확보)",
                    deadline_seconds,
                    len(best),
                    len(TRAITS),
                )
                return best, longest
            LOGGER.warning(
                "근거 파싱이 %d/%d trait에 그쳐 min_new_tokens=%d로 재생성합니다 "
                "(completion_chars=%d)",
                len(best),
                len(TRAITS),
                retry_min_tokens,
                len(completion),
            )
            retried, completion = run(min_new_tokens=retry_min_tokens)
            if len(completion) > len(longest):
                longest = completion
            if len(retried) > len(best):
                best = retried
            if len(best) == len(TRAITS):
                return best, longest
            # --- 2차 복구: 표본추출로 실제로 다른 토큰열을 만든다 ------------------
            # completion이 비어 있지 않은데 JSON이 깨진 경우 greedy 재시도는 **같은
            # 토큰**을 그대로 되돌려주므로 아무 것도 복구하지 못한다. 여기서만 결정적
            # 시드를 걸고 표본추출한다.
            if not budget_left():
                LOGGER.warning(
                    "근거 예산 %.1f초를 소진해 표본추출 재시도를 생략합니다 "
                    "(%d/%d trait 확보)",
                    deadline_seconds,
                    len(best),
                    len(TRAITS),
                )
                return best, longest
            LOGGER.warning(
                "greedy 재시도로도 %d/%d에 그쳐 표본추출로 한 번 더 시도합니다",
                len(best),
                len(TRAITS),
            )
            torch.manual_seed(_SAMPLED_RETRY_SEED)
            sampled, sampled_completion = run(
                do_sample=True,
                temperature=_SAMPLED_RETRY_TEMPERATURE,
                top_p=_SAMPLED_RETRY_TOP_P,
                min_new_tokens=retry_min_tokens,
            )
            if len(sampled_completion) > len(longest):
                longest = sampled_completion
            if len(sampled) > len(best):
                best = sampled
            # 세 개를 다 못 채웠어도 **얻은 만큼 돌려준다.** 못 채운 trait은 respond()가
            # 원문(longest)을 정제해 메우고, 그것도 비면 template으로 간다. 점수는
            # 어느 경우에도 그대로 응답에 실린다.
            return best, longest

    def _rationale_prompt_ids(
        self, text: OfficialRequestText, scores: dict[str, float]
    ) -> tuple[list[int], int]:
        """예산 안에 들어가는 prompt 토큰열과 잘라낸 essay 문자 수를 만든다.

        예산을 넘으면 예전에는 예외를 던져 근거를 포기했고, 그 예외가 `respond()`를 타고
        HTTP 500이 되어 **이미 계산된 점수까지** 함께 버려졌다. essay 뒷부분을 잘라 앞부분만
        보고 쓴 근거는 근거로서 불완전할 뿐이지만, 응답을 잃으면 그 에세이의 RMSE·Spearman
        기여가 통째로 0이 된다.
        """

        spec = self.config.rationale
        tokenizer = self._rationale_tokenizer
        budget = spec.max_length - spec.max_new_tokens

        def encode(source: OfficialRequestText) -> list[int]:
            # 학습과 완전히 같은 message + chat template을 쓴다. 손으로 tokenize하면
            # 어댑터가 학습 때 본 generation prompt 토큰을 못 받아 형식 준수가 무너진다.
            messages = build_rationale_messages(
                source,
                scores,
                prompt_template=spec.rationale_prompt_text,
                # 필드가 없는 옛 spec/진단 fake는 역사적 기본 hint로 해석한다.
                # 실제 `RationaleSpec`은 항상 이 값을 가진다.
                skeleton_hint=getattr(
                    spec, "rationale_skeleton_hint", DEFAULT_SKELETON_HINT
                ),
            )
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **dict(spec.chat_template_kwargs or {}),
            )
            if hasattr(ids, "input_ids"):
                ids = ids.input_ids
            elif isinstance(ids, dict):
                ids = ids["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.reshape(-1).tolist()
            return list(ids)

        prompt_ids = encode(text)
        if len(prompt_ids) <= budget:
            return prompt_ids, 0

        original_chars = len(text.essay)
        keep = original_chars
        for _ in range(_BUDGET_TRUNCATION_ATTEMPTS):
            ratio = budget / max(1, len(prompt_ids))
            keep = max(1, int(keep * ratio * _BUDGET_TRUNCATION_MARGIN))
            candidate = dataclasses.replace(text, essay=text.essay[:keep])
            prompt_ids = encode(candidate)
            if len(prompt_ids) <= budget:
                return prompt_ids, original_chars - keep
        # 여기까지 왔으면 essay가 아니라 prompt/template 자체가 예산을 넘는 것이다.
        # 그래도 응답은 만들어야 하므로 토큰열 앞부분만 남긴다.
        LOGGER.error(
            "essay 절단으로도 예산 %d를 못 맞춰 prompt 토큰을 직접 자릅니다 (len=%d)",
            budget,
            len(prompt_ids),
        )
        return prompt_ids[:budget], original_chars - keep

    # --- full response ---------------------------------------------------
    def respond(self, text: OfficialRequestText) -> tuple[str, dict[str, Any]]:
        """서버가 그대로 돌려줄 JSON 문자열과 진단 정보를 만든다.

        **이 함수는 점수를 버리지 않는다.** 근거 생성이 어떤 이유로 실패해도 이미 확정된
        정수 점수를 그대로 응답에 싣고, 채우지 못한 근거만 template으로 메운다. 점수 경로
        자체가 죽은 경우에만 중앙 상수 점수로 강등한다.

        예전 구현은 근거가 비면 ``RuntimeError``를 올렸고 그것이 HTTP 500이 되어 평가
        서버에서 **그 에세이가 0점으로 집계**됐다. validation gold 기준 0점 한 행의
        제곱오차는 11.97, 상수 점수를 낸 행은 0.43이다. 400편에서 3편만 그렇게 잃어도
        mean-first RMSE가 0.4168 -> 0.5118로 무너진다. 근거 품질(LLM Judge, 가중치 10%)을
        조금 잃는 것과는 비교가 되지 않는 손실이므로 정책을 뒤집었다. 강등은 전부
        ``degradation``과 서버 로그에 남는다.

        규정 §5의 병렬 요청을 순차로 직렬화한다. 48GB에서 동시 요청이 KV cache를 겹쳐 쌓아
        OOM으로 전체 응답을 잃는 것보다 순차가 안전하며 규정이 순차를 명시적으로 허용한다.
        """

        degradation: list[str] = []
        raw_scores: dict[str, float] | None = None
        # 락 대기 시간을 남긴다. 평가 서버가 동시 요청을 보내면 이 값이 곧 큐 길이이고,
        # 그들의 요청 timeout을 넘기면 우리 잘못 없이 그 에세이가 0점이 된다. 우리가 그들의
        # timeout·동시성을 모르므로 최소한 사후에 셀 수 있게 기록만 한다.
        queued_at = time.monotonic()
        with self._inference_lock:
            lock_wait = time.monotonic() - queued_at
            # 락 **안**에서 초기화한다. 밖에서 하면 뒤이어 들어온 요청이 앞 요청의
            # 생성 도중에 카운터를 0으로 되돌려 진단이 거짓말을 한다.
            self._deadline_hits = 0
            if lock_wait > _LOCK_WAIT_WARN_SECONDS:
                LOGGER.warning(
                    "직렬화 큐에서 %.1f초 대기했습니다. 평가 서버가 동시 요청을 보내고 "
                    "있다면 요청 timeout 위험 구간입니다.",
                    lock_wait,
                )
            try:
                # 미학습 문항이면 세 영역에 같은 상수를 더한다. **정수화 전에**
                # 해야 합의 반올림이 실제로 움직인다. 본 문항이면 그대로다.
                raw_scores = _finite_scores(self.score(text))
                # **반올림을 근거 생성보다 먼저 한다.** 응답에 실리는 점수는 정수
                # 삼중이므로 근거도 그 정수를 정당화해야 한다. 예전에는 실수 원본을 근거
                # 모델에 넘기고 반올림을 그 뒤에 해서, 근거는 3.4를 설명하는데 출력 점수는
                # 3이 되는 불일치가 있었다. LLM Judge가 점수-근거 정합성을 직접 본다.
                submitted = _submitted_integer_scores(
                    raw_scores, postprocessor=self._score_postprocessor
                )
            except Exception:  # noqa: BLE001 - 응답을 잃는 것보다 상수 점수가 낫다
                LOGGER.exception(
                    "점수 경로가 실패해 중앙 상수 점수 %s로 강등합니다",
                    NEUTRAL_SUBMITTED_SCORES,
                )
                degradation.append("score_failed")
                submitted, rationale_map = last_resort_outputs()
                substituted = list(TRAITS)
            else:
                generated: dict[str, str] = {}
                raw_completion = ""
                if not self.config.rationale.enabled:
                    degradation.append("rationale_disabled")
                else:
                    try:
                        generated, raw_completion = self.rationales_with_raw(
                            text, submitted
                        )
                    except Exception:  # noqa: BLE001 - 점수는 이미 확정되어 있다
                        LOGGER.exception(
                            "근거 경로가 실패했습니다. 확정 점수 %s는 그대로 싣고 "
                            "template 근거로 응답합니다",
                            submitted,
                        )
                        degradation.append("rationale_failed")
                try:
                    rationale_map, substituted = fill_missing_rationales(
                        generated, submitted, raw_completion=raw_completion
                    )
                except Exception:  # noqa: BLE001 - 여기서 죽으면 진짜 점수를 잃는다
                    # 근거를 **채우는** 단계까지 실패하는 것은 예상 밖이지만, 그때
                    # 예외를 올리면 이미 확정된 정수 점수가 중립 3점으로 떨어진다.
                    # 근거만 상수로 낮추고 점수는 지킨다.
                    LOGGER.exception(
                        "근거 채우기가 실패해 상수 근거로 대체합니다. 점수 %s는 유지",
                        submitted,
                    )
                    rationale_map = {trait: FALLBACK_RATIONALE for trait in TRAITS}
                    substituted = list(TRAITS)
                    degradation.append("rationale_fill_failed")
                if substituted and not degradation:
                    # 이미 상위 원인(생성 실패/비활성)이 기록됐으면 중복해서 남기지 않는다.
                    degradation.append(
                        "rationale_substituted:" + ",".join(substituted)
                    )
        # 근거 dict가 어떤 이유로든 불완전하면 그 자리에서 메운다. KeyError 하나가
        # 확정된 점수를 통째로 날리는 경로를 남기지 않는다.
        outputs = {
            trait: TraitOutput(
                score=submitted[trait],
                rationale=(rationale_map.get(trait) or FALLBACK_RATIONALE),
            )
            for trait in TRAITS
        }
        # 나갈 문자열을 공식 파서에 직접 먹여 보고 통과할 때만 반환한다.
        # 통과 못 하면 점수를 지킨 채 근거만 낮춰 다시 만든다.
        body, parse_guard = enforce_official_parse(outputs)
        if parse_guard["repaired"]:
            degradation.append("parse_guard:" + parse_guard["final_stage"])
        return body, {
            "parse_guard": parse_guard,
            "scores": raw_scores if raw_scores is not None else dict(submitted),
            "submitted_scores": submitted,
            "member_count": len(self._members),
            "lock_wait_seconds": lock_wait,
            # 근거 생성이 벽시계 예산에 걸려 잘린 횟수. 0이 아니면 그 편은 근거를
            # 일부 잃었지만 **점수는 온전히 실렸다**. 배포 게이트가 이 값을 센다.
            "rationale_deadline_hits": int(getattr(self, "_deadline_hits", 0)),
            "degradation": degradation,
            "substituted_rationales": substituted,
            "rationale_error": degradation[0] if degradation else None,
            "rationale_lengths": {
                trait: len(outputs[trait].rationale) for trait in TRAITS
            },
        }


def _finite_scores(scores: dict[str, float]) -> dict[str, float]:
    """모델이 NaN/inf를 내면 그 trait만 중앙값으로 바꾼다.

    ``build_response_json``의 ``clamp_score``는 비유한 값에 예외를 던진다. 그 예외가
    그대로 올라가면 응답 전체를 잃으므로 여기서 미리 막는다.
    """

    cleaned: dict[str, float] = {}
    for trait in TRAITS:
        value = float(scores[trait])
        if not math.isfinite(value):
            LOGGER.error(
                "%s 점수가 %r로 비유한 값이라 %.1f로 대체합니다",
                trait,
                value,
                NEUTRAL_SUBMITTED_SCORES[trait],
            )
            value = NEUTRAL_SUBMITTED_SCORES[trait]
        cleaned[trait] = value
    return cleaned


def _submitted_integer_scores(
    scores: dict[str, float],
    *,
    postprocessor: ScorePostprocessor = _POSTPROCESSOR,
) -> dict[str, float]:
    """소수 점수를 제출용 정수 삼중으로 바꾼다.

    연구 파이프라인의 `infer`와 **같은 class**를 부른다. 예전에는 이 함수가 변환을
    직접 구현하고 `infer`는 변환을 아예 하지 않아, 같은 checkpoint의 연구 산출물과
    서빙 출력이 `.4249` vs `.4477`로 갈라져 있었다.
    """

    return postprocessor.apply_row(scores)


def parse_rationale_completion(completion: str) -> dict[str, str]:
    """근거 모델 출력에서 얻을 수 있는 trait 근거를 **얻은 만큼** 꺼낸다.

    먼저 strict JSON으로 읽고, 본문 인용부호 하나를 escape하지 않은 실측 오류만 고정
    C/O/E shape에서 회수한다.

    예전에는 세 근거가 모두 차지 않으면 빈 dict로 fail-closed했다. 그러면 `respond()`가
    예외를 올리고 HTTP 500이 되어 **이미 확정된 점수까지 함께 버려졌다.** 이제는 확보한
    trait만 돌려주고, 못 채운 trait은 호출자가 재생성하거나 template으로 메운다. 점수는
    어느 경로에서도 응답에 그대로 실린다. 각 trait 값 자체의 검증(문자열, 비어 있지 않음,
    학습 template placeholder가 아님)은 예전과 똑같이 엄격하다.
    """

    from .schema import _extract_first_json

    # 중괄호가 깨지면 `_extract_first_json`이 None을 낸다. 예전에는 여기서 곧장
    # 빈 dict를 돌려줘 아래 회수 경로에 **도달하지 못했다.** 구조를 못 읽어도 원문
    # 스캔은 여전히 가능하므로 조기 반환하지 않는다.
    candidate = _extract_first_json(completion) or ""
    try:
        parsed = json.loads(candidate) if candidate else None
    except json.JSONDecodeError:
        parsed = None
    found: dict[str, str] = {}
    if isinstance(parsed, dict):
        for trait in TRAITS:
            value = parsed.get(trait)
            if isinstance(value, dict):
                value = value.get("rationale")
            if not isinstance(value, str) or not value.strip():
                continue
            cleaned = value.strip()
            if _contains_template_placeholder(cleaned):
                # 학습 template 문구를 그대로 뱉은 trait은 근거가 아니다. 그 trait만
                # 버리고 나머지는 살린다.
                continue
            found[trait] = cleaned
    if len(found) == len(TRAITS):
        return found
    # 모델이 본문의 인용부호를 JSON escape하지 않은 단일 실측 사례가 있었다. 점수는
    # 생성 JSON을 신뢰하지 않고 score head 값을 별도로 복사하므로, 여기서는 고정된
    # C/O/E trait shape의 rationale 문자열만 엄격하게 회수한다. 구조 회수는 세 개를
    # 한꺼번에 맞출 때만 신뢰할 수 있으므로 이 함수만 all-or-nothing으로 남긴다.
    salvaged = _salvage_fixed_shape_rationales(candidate) if candidate else {}
    if len(salvaged) > len(found):
        LOGGER.warning(
            "escape되지 않은 인용부호가 있는 세 근거를 고정 shape에서 회수했습니다"
        )
        return salvaged
    if len(found) < len(TRAITS):
        # 마지막 회수: JSON 구조를 아예 포기하고 완성문 **원문 전체**에서 trait별
        # rationale 값만 긁는다. 중괄호가 깨져 `_extract_first_json`이 엉뚱한 데서
        # 잘렸을 때도 동작한다. 여기서 얻는 근거가 template이나 빈 문자열이면 그
        # trait은 여전히 버리고 호출자가 메운다.
        loose = _loose_scan_rationales(completion)
        merged = {**loose, **found}   # strict로 얻은 값이 항상 우선한다
        if len(merged) > len(found):
            LOGGER.warning(
                "JSON 구조 회수에 실패해 원문 스캔으로 근거 %d개를 얻었습니다",
                len(merged) - len(found),
            )
        return merged
    return found


_LOOSE_RATIONALE_RE = {
    trait: re.compile(
        rf'"{trait}"\s*:\s*\{{.*?"rationale"\s*:\s*"(?P<body>.*?)"\s*[,}}]',
        re.S,
    )
    for trait in TRAITS
}


def _loose_scan_rationales(completion: str) -> dict[str, str]:
    """JSON 파싱을 포기하고 원문에서 trait별 rationale만 정규식으로 긁는다.

    구조를 신뢰하지 않으므로 trait 단위로 독립 회수한다. 하나도 못 얻으면 빈 dict다.
    점수는 이 경로를 절대 거치지 않는다. 점수는 score head 값을 그대로 쓰고 생성
    JSON을 신뢰하지 않는다.
    """

    found: dict[str, str] = {}
    for trait, pattern in _LOOSE_RATIONALE_RE.items():
        match = pattern.search(completion or "")
        if match is None:
            continue
        decoded = _decode_salvaged_json_string(match.group("body"))
        if decoded is None:
            continue
        cleaned = decoded.strip()
        if not cleaned or _contains_template_placeholder(cleaned):
            continue
        found[trait] = cleaned
    return found


def _salvage_fixed_shape_rationales(candidate: str) -> dict[str, str]:
    """깨진 JSON에서 고정 C/O/E 순서의 rationale 세 개만 fail-closed 회수한다.

    허용하는 오류는 rationale 값 안의 escape되지 않은 큰따옴표뿐이다. 각 trait marker와
    ``score``/``rationale`` marker가 정확히 한 번, 정해진 순서로 있어야 하고 각 segment는
    trait 객체의 닫는 중괄호로 끝나야 한다. 이보다 느슨하면 본문 문자열을 구조로 오인할 수
    있으므로 빈 dict를 돌려준다.
    """

    if not candidate.startswith("{") or not candidate.endswith("}"):
        return {}

    trait_matches: list[re.Match[str]] = []
    for trait in TRAITS:
        pattern = re.compile(rf'"{re.escape(trait)}"\s*:\s*\{{')
        matches = list(pattern.finditer(candidate))
        if len(matches) != 1:
            return {}
        trait_matches.append(matches[0])
    if [match.start() for match in trait_matches] != sorted(
        match.start() for match in trait_matches
    ):
        return {}
    if candidate[1 : trait_matches[0].start()].strip():
        return {}

    found: dict[str, str] = {}
    for index, (trait, marker) in enumerate(zip(TRAITS, trait_matches, strict=True)):
        end = (
            trait_matches[index + 1].start()
            if index + 1 < len(trait_matches)
            else len(candidate)
        )
        segment = candidate[marker.start() : end]
        if len(re.findall(r'"score"\s*:', segment)) != 1:
            return {}
        rationale_markers = list(re.finditer(r'"rationale"\s*:\s*"', segment))
        if len(rationale_markers) != 1:
            return {}
        closing = re.search(
            r'"\s*\}\s*,\s*$' if index + 1 < len(TRAITS) else r'"\s*\}\s*\}\s*$',
            segment,
        )
        if closing is None:
            return {}
        raw = segment[rationale_markers[0].end() : closing.start()]
        decoded = _decode_salvaged_json_string(raw)
        if decoded is None or not decoded.strip():
            return {}
        cleaned = decoded.strip()
        if _contains_template_placeholder(cleaned):
            return {}
        found[trait] = cleaned
    return found if len(found) == len(TRAITS) else {}


def _decode_salvaged_json_string(raw: str) -> str | None:
    """문자열 내부의 escape되지 않은 큰따옴표만 escape한 뒤 JSON unescape한다."""

    escaped: list[str] = []
    backslashes = 0
    for char in raw:
        if char == '"' and backslashes % 2 == 0:
            escaped.append("\\")
        escaped.append(char)
        if char == "\\":
            backslashes += 1
        else:
            backslashes = 0
    try:
        value = json.loads('"' + "".join(escaped) + '"')
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None
