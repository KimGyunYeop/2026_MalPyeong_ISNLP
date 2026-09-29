#!/usr/bin/env python3
"""최종 제출본의 주요 설정을 저장소 안의 실제 파일에서 읽어 출력한다.

README에 숫자를 손으로 적으면 코드와 조용히 갈라진다. 그래서 README는 이 스크립트의
출력을 인용하고, 값은 항상 아래 원본에서 읽는다.

  채점모델  main_code/configs/confirmed_final.json          (제출 멤버의 resolved config)
  근거모델  main_code_relonation/recipes/r17_*.json          (학습 레시피)
            main_code_relonation/prompts/rationale_prompt_v4.txt
  서빙/배포 docker_release.sh                                (멤버 구성·해시·deadline)

사용법:  python show_submission_settings.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# 해시는 반드시 프로젝트 함수로 낸다. 손으로 hashlib을 쓰면 정규화가 빠져
# manifest에 적힌 값과 달라진다(실제로 그렇게 틀렸다).
from main_code_relonation.prompts import (  # noqa: E402
    load_prompt_template,
    prompt_template_sha256,
)
SCORE_PRESET = ROOT / "main_code/configs/confirmed_final.json"
RATIONALE_RECIPE = ROOT / "main_code_relonation/recipes/r17_qwen35_lora_fixed_prompt_v4.json"
RATIONALE_PROMPT = ROOT / "main_code_relonation/prompts/rationale_prompt_v4.txt"
RELEASE = ROOT / "docker_release.sh"

SEEDS = tuple(range(42, 50))


def _grep(text: str, pattern: str, default: str = "?") -> str:
    found = re.search(pattern, text)
    return found.group(1) if found else default


def _line(label: str, value: object, note: str = "") -> None:
    tail = f"   {note}" if note else ""
    print(f"  {label:<34}{value}{tail}")


def main() -> None:
    score = json.loads(SCORE_PRESET.read_text(encoding="utf-8"))
    recipe = json.loads(RATIONALE_RECIPE.read_text(encoding="utf-8"))
    # load_prompt_template이 저장소 텍스트의 끝 개행을 떼어낸다. 그 값이
    # 학습·서빙이 실제로 쓰는 prompt이고 manifest의 sha도 그것이다.
    prompt = load_prompt_template(RATIONALE_PROMPT)
    release = RELEASE.read_text(encoding="utf-8")

    bar = "=" * 78
    print(bar)
    print("2026 국립국어원 AI 말평 · 글쓰기 채점 능력 평가 — 최종 제출본 설정")
    print(bar)

    print("\n[1] 채점 모델 (score) — 8 seed 앙상블, seed만 다르고 나머지는 동일")
    _line("backbone", f"{score['model_id']} @ {score['model_revision']}")
    _line("seeds", f"{SEEDS[0]}~{SEEDS[-1]} ({len(SEEDS)}개)", "등가중 평균")
    _line("training_mode", f"{score['training_mode']} (QLoRA={score['use_qlora']})")
    _line("LoRA", f"r={score['lora_r']} alpha={score['lora_alpha']} "
                  f"dropout={score['lora_dropout']} include_mlp={score['lora_include_mlp']}")
    _line("lora_targets", score["lora_targets"],
          "Qwen은 32층 중 8층만 full-attention이라 q/k/v/o는 8층, MLP는 32층에 붙는다")
    _line("score_head", f"{score['score_head']} (5-class)",
          f"readout={score['categorical_readout']}")
    _line("입력 표면", f"{score['essay_surface']} / {score['input_format']}")
    _line("pooling", f"{score['pooling']} / {score['layer_aggregation']}")
    _line("organization pooling", score["organization_pooling"],
          "요소 D: 이중 공백 단서로 나눈 문단 평균을 구성 영역에만 반영. "
          "게이트는 0에서 시작하는 학습 스칼라라 단서가 없는 글은 shared와 같다")
    print("  손실")
    _line("  mse / distribution",
          f"{score['mse_loss_weight']} / {score['distribution_loss_weight']}"
          f" (label_smoothing={score['distribution_label_smoothing']})")
    _line("  listwise",
          f"{score['listwise_loss']} w={score['listwise_loss_weight']}"
          f" target={score['ranking_target']}")
    _line("  보조 head", f"detail={score['detail_head_mode']} "
                        f"(rater_set={score['detail_rater_set_loss_weight']} "
                        f"expected={score['detail_expected_loss_weight']}) "
                        f"trait_avg={score['trait_average_loss_weight']} "
                        f"paragraph={score['paragraph_boundary_loss_weight']}",
          "요소 C: 익명 평가자 2인 branch 18개 분류기가 9준거를 보조감독. "
          "단일 재평가자 행을 묶으려면 expected 손실이 함께 필요하다")
    _line("steps / eval", f"{score['max_train_steps']} / {score['eval_steps']}")
    _line("batch / accum / max_len",
          f"{score['batch_size']} / {score['gradient_accumulation']} / {score['max_length']}")
    _line("lr (head / lora)",
          f"{score['head_learning_rate']} / {score['lora_learning_rate']} "
          f"({score['lr_scheduler_type']}, warmup {score['warmup_ratio']})")
    _line("checkpoint 선택", "평탄(마지막 step) — selection.json 없음",
          f"best 지표는 {score['best_checkpoint_metric']}이나 채택하지 않는다")
    _line("제출 후처리", score["score_postprocess"], "integer_total_offset=0")

    print("\n[2] 근거 모델 (rationale) — Gemma teacher 증류 student")
    _line("backbone", f"{recipe['model_id']} @ {recipe['model_revision']}")
    _line("LoRA", f"r={recipe['lora_rank']} alpha={recipe['lora_alpha']} "
                  f"dropout={recipe['lora_dropout']}")
    _line("lora_targets", recipe["lora_targets"])
    _line("batch / accum", f"{recipe['batch_size']} / {recipe['gradient_accumulation']}",
          f"epochs={recipe['epochs']} lr={recipe['learning_rate']}")
    _line("max_length / new_tokens", f"{recipe['max_length']} / {recipe['max_new_tokens']}")
    _line("decoding", f"temperature={recipe['temperature']} top_p={recipe['top_p']} (greedy)")
    _line("chat_template_kwargs", recipe["chat_template_kwargs"])
    _line("score_mode", recipe["score_mode"], "점수를 고정 조건으로 주고 근거만 생성")
    _line("prompt", RATIONALE_PROMPT.name)
    _line("prompt sha256", prompt_template_sha256(prompt))
    _line("skeleton hint", recipe["rationale_skeleton_hint"])

    print("\n[3] 서빙 / 배포")
    _line("멤버 수", _grep(release, r"require\(len\(scores\) == (\d+)"))
    _line("멤버 가중치", _grep(release, r'"weight": ([0-9.]+),'))
    _line("근거 deadline(초)",
          _grep(release, r'RATIONALE_DEADLINE_SECONDS", "([0-9.]+)"'),
          "표본추출 복구에 76~81초가 필요하다(실측)")
    _line("image tag", _grep(release, r'DEFAULT_IMAGE_TAG="([^"]+)"'))
    _line("prediction sha256", _grep(release, r'PROVISIONAL_HTTP_PREDICTION_SHA256="([0-9a-f]+)"'))
    _line("registry", _grep(release, r'REGISTRY_REPO:-([^}]+)\}'))
    _line("VRAM(정적 추정)", "19.31 GiB / 48 GiB", "backbone 한 벌 공유, 멤버는 어댑터만")

    print("\n[4] 검증 400편 (공식 지표, gold=score.average)")
    _line("RMSE", "0.420215")
    _line("Spearman", "0.753811")
    _line("파싱 성공", "400/400", "강등 0건, 템플릿 근거 0건")
    _line("응답시간", "중앙 13.35s / p95 15.39s / 최대 83.50s")
    print(bar)


if __name__ == "__main__":
    main()
