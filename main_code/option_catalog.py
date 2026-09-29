"""학습 설정 옵션 카탈로그를 **코드에서** 생성한다 (2026-08-21).

왜 생성기인가
------------
사람이 기억으로 적은 옵션 목록은 곧 틀린다. 2026-08-21에 실제로 그랬다 — "어제 구현한
축"을 기억으로 나열했는데 `RegressionConfig`는 필드가 152개이고, 그중 52개는 한 번도
기본값을 벗어난 적이 없었다. 그래서 이 파일은 목록을 **코드와 실험 산출물에서 뽑는다.**

  옵션 이름·타입·기본값   ← `RegressionConfig` dataclass
  CLI 플래그              ← `train.py --help` 실제 출력
  시험 이력               ← `results/**/resolved_category.json`의 값 집합
  분할 여부               ← `unseen_prompt_holdout`이 비어 있지 않은 run

분류는 이 파일의 `CATEGORY_RULES`에 있다. 새 옵션을 추가하면 규칙에 걸리지 않아
`미분류`로 나오므로, 그때 규칙을 갱신한다(조용히 빠지지 않게 하려는 의도).

쓰는 법
------
    python -m main_code.option_catalog                  # 마크다운을 stdout에
    python -m main_code.option_catalog --output <경로>   # 파일로
    python -m main_code.option_catalog --only-untested   # 미시험 축만
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "main_code/results"
# 데이터 분할(LOPO) 실험과 전체 재학습 실험을 나눠 기록한다 (2026-08-21 규약).
SPLIT_TREE = "final_proposed_datasplited"
FULL_TREE = "final_proposed"

# ---------------------------------------------------------------- 분류 규칙
# (분류, 정규식). 위에서부터 먼저 맞는 것을 쓴다. 순서가 의미를 가진다.
CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("점수 후처리", r"^(score_postprocess)$"),
    ("평가·checkpoint 선택", r"(checkpoint_metric|checkpoint_retention|^eval_steps$|^eval_every_epoch$|^schema_version$)"),
    ("데이터 분할·보류", r"^(unseen_prompt_holdout|validation_holdout_size|allow_validation_\w+|split_primary_sources)$"),
    ("데이터 선택·혼합", r"^(dataset_\w+|primary_data_profile|competition_\w+|extended_\w+|external_(?!score_alignment)\w+|origin_extra_\w+|initial_lora_adapter|final_competition_epochs)$"),
    ("전처리·입력 표면", r"^(essay_surface|input_format|max_length|train_canonical_surface_probability|rubric_profile|prompt_registry|organization_augmentation|detail_rater_registry)$"),
    ("라벨 가공", r"^(train_label_rounding|distribution_label_smoothing|external_score_alignment|external_organization_label_policy)$"),
    ("백본·LoRA", r"^(model_\w+|backbone_type|trust_remote_code|use_qlora|gradient_checkpointing|lora_\w+|training_mode)$"),
    ("구조: pooling·표현", r"^(pooling|organization_pooling|layer_aggregation|last_n_layers|normalize_features|pooled_normalization|attention_pool_dim|analysis_pool_\w+)$"),
    ("구조: head", r"^(head_type|head_hidden_size|score_head|score_parameterization|categorical_readout|criterion_readout|detail_head_mode|detail_final_source|prompt_head_\w+|mixed_head_weight|ordinal_steps|ordinal_blend_weight|prompt_adversary_classes|level_prototype_(count|momentum|temperature))$"),
    ("알고리즘: 양자화 인식", r"^(quantization_\w+|quantized_\w+)$"),
    ("알고리즘: 순위 학습", r"^(pairwise_\w+|listwise_\w+|ranking_\w+|trait_average_pairwise_weight)$"),
    ("알고리즘: 분포 이동·강건화", r"^(tail_weight_\w+|group_dro_step_size|level_prototype_weight|prompt_dropout_probability|prompt_adversary_weight)$"),
    ("손실 가중", r"(loss_weight$|^mse_loss_weight$|^distribution_loss_weight$|^trait_average_loss_weight$|^gumbel_temperature$)"),
    ("학습 스케줄", r"^(batch_size|gradient_accumulation|epochs|head_epochs|joint_epochs|max_train_steps|head_warmup_steps|warmup_ratio|lr_scheduler_type|lora_scheduler_scope|weight_decay|max_grad_norm|seed|head_learning_rate|quantized_loss_start_step|quantized_loss_ramp_steps|quantization_anneal_steps)$"),
    # batch 구성은 순위 손실이 무엇을 비교하는지를 정하므로 순위 학습으로 분류한다.
    ("알고리즘: 순위 학습", r"^(inbatch_sampling)$"),
    ("경로", r"^(dataset_root|extended_data_dir)$"),
)

# 같은 분류에 규칙이 여러 개 있을 수 있으므로 순서를 지키며 중복을 없앤다.
CATEGORY_ORDER = list(dict.fromkeys(name for name, _ in CATEGORY_RULES)) + ["미분류"]


def categorize(name: str) -> str:
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, name):
            return category
    return "미분류"


# ------------------------------------------------------------------ CLI 플래그


def cli_flags() -> set[str]:
    """`train.py --help`에서 실제 플래그를 뽑는다. 추측하지 않는다."""

    try:
        out = subprocess.run(
            [sys.executable, "-m", "main_code.train", "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
            env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
        ).stdout
    except Exception:
        return set()
    return set(re.findall(r"--[a-z0-9][a-z0-9-]*", out))


def flag_for(name: str, kind: Any, flags: set[str]) -> str:
    """필드 이름에 대응하는 CLI 표기를 만든다. bool은 --x / --no-x 쌍을 확인한다."""

    dashed = name.replace("_", "-")
    plain, negated = f"--{dashed}", f"--no-{dashed}"
    if kind is bool:
        if plain in flags and negated in flags:
            return f"`{plain}` / `{negated}`"
        if negated in flags:
            return f"`{negated}`"
    if plain in flags:
        return f"`{plain}`"
    return f"`{plain}` <sup>(플래그 미확인)</sup>"


# ------------------------------------------------------------------ 시험 이력


def normalize(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def load_configs() -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in RESULTS.rglob("resolved_config.json"):
        text = str(path)
        if "_ABANDONED" in text or "_SUPERSEDED" in text:
            continue
        try:
            out.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


def field_defaults() -> dict[str, Any]:
    from .config import RegressionConfig

    out: dict[str, Any] = {}
    for field in dataclasses.fields(RegressionConfig):
        if field.default is not dataclasses.MISSING:
            out[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                out[field.name] = field.default_factory()  # type: ignore[misc]
            except Exception:
                out[field.name] = None
        else:
            out[field.name] = None
    return out


def short(value: Any, limit: int = 34) -> str:
    text = "(빈 문자열)" if value == "" else str(value)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "\\|")


def build_rows() -> list[dict[str, Any]]:
    from .config import RegressionConfig

    flags = cli_flags()
    defaults = field_defaults()
    configs = load_configs()
    split = [(p, d) for p, d in configs if d.get("unseen_prompt_holdout")]
    kinds = {f.name: f.type for f in dataclasses.fields(RegressionConfig)}

    rows: list[dict[str, Any]] = []
    for field in dataclasses.fields(RegressionConfig):
        name = field.name
        default = normalize(defaults.get(name))
        everywhere: Counter = Counter()
        in_split: Counter = Counter()
        for _, data in configs:
            if name in data:
                everywhere[normalize(data[name])] += 1
        for _, data in split:
            if name in data:
                in_split[normalize(data[name])] += 1
        non_default = sorted((v for v in everywhere if v != default), key=str)
        non_default_split = sorted((v for v in in_split if v != default), key=str)
        if non_default_split:
            status, mark = "분할에서 시험", "✅"
        elif non_default:
            status, mark = "전체에서만 시험", "🔶"
        else:
            status, mark = "미시험", "❌"
        kind = kinds.get(name)
        rows.append(
            dict(
                name=name,
                category=categorize(name),
                flag=flag_for(name, bool if kind is bool or kind == "bool" else kind, flags),
                type=getattr(kind, "__name__", str(kind)).replace("builtins.", ""),
                default=default,
                tested=non_default,
                tested_split=non_default_split,
                status=status,
                mark=mark,
            )
        )
    return rows


# ------------------------------------------------------------------- 렌더링

DESCRIPTIONS: dict[str, str] = {
    # 이 표의 목적은 "이 옵션이 무엇을 바꾸는가"를 한 줄로 남기는 것이다.
    # 비어 있으면 이름과 분류만 남는다. 채워 넣을수록 표가 쓸모 있어진다.
    "unseen_prompt_holdout": "지정 문항을 **학습에서만** 제외한다. 평가 400편은 그대로라 내부/외부가 동시에 나온다",
    "prompt_dropout_probability": "학습 중 확률적으로 지문을 지운다. 지문 의존을 줄여 미학습 문항 일반화를 노린다",
    "prompt_adversary_weight": "gradient reversal로 문항을 못 맞히게 만든다(DANN). 문항 불변 표현을 노린다",
    "pooled_normalization": "pooled 표현에 parameterless LayerNorm. 표현 크기 폭주를 줄인다",
    "group_dro_step_size": "문항별 손실이 큰 그룹의 가중치를 지수 상승(GroupDRO). 최악 문항을 직접 최적화",
    "level_prototype_weight": "수준 bin별 EMA prototype에 대한 prototypical cross-entropy (PLAES 계열)",
    "max_grad_norm": "gradient clipping 임계. c02는 1.0이었고 실측 grad_norm이 5.8~19.7이었다",
    "tail_weight_target_sigma": "라벨 값으로 표본 가중을 미리 정한다. 전제(숨은 SD 확대)가 무너져 폐기",
    "quantization_rule": "학습 중 예측을 제출 표면으로 양자화하는 규칙",
    "quantization_surrogate": "양자화의 미분 가능 대체(soft / straight_through / expected_risk)",
    "quantized_trait_loss_weight": "양자화된 영역 점수에 대한 손실 가중",
    "quantized_mean_rank_weight": "양자화된 평균의 순위 손실 가중",
    "score_head": "점수 head 형태(regression / distribution / regression_ordinal 등)",
    "score_parameterization": "영역 3개를 직접 예측할지, 평균+대비로 분해할지",
    "detail_head_mode": "9준거 보조 감독 방식. `rater_set`이 c02에 들어 있고 0.0094 개선했다",
    "detail_final_source": "최종 영역 점수를 head에서 직접 낼지(9준거 평균으로 낼지)",
    "pairwise_loss": "쌍 비교 순위 손실(RankNet 계열)",
    "listwise_loss": "목록 단위 순위 손실(soft Spearman)",
    "trait_average_loss_weight": "세 영역 평균에 직접 거는 손실 가중. 공식 지표가 보는 양이다",
    "best_checkpoint_metric": "checkpoint 선택 저울. 이것이 무엇을 보느냐가 결과를 바꾼다",
    "score_postprocess": "제출 표면 변환. `average_matched`가 세 정수 합을 연속 합에 맞춘다",
    "input_format": "프롬프트 조립 형식. `essay_only_v1`은 지문을 아예 넣지 않는다",
    "essay_surface": "에세이 원문 표면(공백·개행 처리)",
    "seed": "학습 재실행 난수. **실측 재실행 SD가 제출 RMSE 0.0138**이므로 반복이 필수다",
    "lora_r": "LoRA rank. 용량 축",
    "origin_extra_max_rater_disagreement": "평가자 불일치가 큰 표본을 학습에서 제외하는 문턱",
    "train_label_rounding": "학습 라벨을 정수로 반올림할지",
    "organization_augmentation": "문단/문장 순서를 섞어 조직 영역 감독을 만드는 증강",
}


def render(rows: list[dict[str, Any]], *, only_untested: bool = False) -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    counts = Counter(row["status"] for row in rows)
    out = [
        "# 학습 옵션 카탈로그 (코드 생성)",
        "",
        f"- 생성: {stamp}  ·  `python -m main_code.option_catalog`",
        f"- `RegressionConfig` 필드 **{len(rows)}개**",
        f"- ✅ 분할에서 시험 **{counts.get('분할에서 시험', 0)}개** / "
        f"🔶 전체에서만 시험 **{counts.get('전체에서만 시험', 0)}개** / "
        f"❌ 미시험 **{counts.get('미시험', 0)}개**",
        "",
        "> 이 표는 손으로 쓰지 않는다. 옵션 이름·타입·기본값은 dataclass에서, CLI 표기는",
        "> `train.py --help`에서, 시험 이력은 `results/**/resolved_config.json`에서 뽑는다.",
        "> 분류 규칙은 `main_code/option_catalog.py`의 `CATEGORY_RULES`에 있고, 새 옵션이",
        "> 규칙에 안 걸리면 `미분류`로 떠서 조용히 빠지지 않는다.",
        "",
        "**상태 기호** ✅ 분할(LOPO) 설정에서 기본값 아닌 값이 시험됨 · "
        "🔶 전체 데이터에서만 시험 · ❌ 기본값에서 벗어난 적 없음",
        "",
    ]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if only_untested and row["status"] == "분할에서 시험":
            continue
        by_category.setdefault(row["category"], []).append(row)

    for category in CATEGORY_ORDER:
        group = by_category.get(category)
        if not group:
            continue
        out += [
            f"## {category}",
            "",
            "| | 옵션 | CLI | 타입 | 기본값 | 설명 | 전체에서 시험된 값 | 분할에서 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in sorted(group, key=lambda r: (r["status"] != "미시험", r["name"])):
            tested = ", ".join(short(v, 18) for v in row["tested"][:4]) or "—"
            if len(row["tested"]) > 4:
                tested += f" <sup>+{len(row['tested']) - 4}</sup>"
            tested_split = ", ".join(short(v, 14) for v in row["tested_split"][:3]) or "—"
            out.append(
                f"| {row['mark']} | `{row['name']}` | {row['flag']} | {row['type']} | "
                f"{short(row['default'], 22)} | {DESCRIPTIONS.get(row['name'], '')} | "
                f"{tested} | {tested_split} |"
            )
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--only-untested", action="store_true")
    args = parser.parse_args(argv)
    text = render(build_rows(), only_untested=args.only_untested)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"{args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
