#!/usr/bin/env bash
# 근거 생성 어댑터 end-to-end 파이프라인.
#
# 흐름은 네 단계다. 각 단계는 독립 실행 가능하며 산출물이 다음 단계 입력이 된다.
#
#   0) teacher   대형 instruct 모델을 vLLM OpenAI 호환 서버로 띄운다.
#   1) generate  **확정 점수를 조건으로** teacher가 근거 JSON을 생성한다(pseudo label).
#   2) judge     (선택) 별도의 공통 proxy Judge가 형식/정합성 실패를 버린다.
#                기술서 학습에는 쓰지 않았다. 사용하려면 train에 PSEUDO_FILE로 그 출력을 넘긴다.
#   3) train     generate 출력 중 파싱·점수 복사·QC를 통과한 행으로 새 LoRA를 학습한다.
#                점수는 절대 재생성하지 않는다.
#   4) verify    학습 어댑터로 추론해 형식 준수와 점수 보존을 확인한다.
#
# ## 왜 점수를 고정하는가
#
# 정량 지표(RMSE 45% + Spearman 45%)는 점수 head가 만든다. 근거 모델이 점수를 다시 만들면
# 그 90%가 근거 모델 품질에 종속된다. 그래서 `score_mode=fixed`만 쓴다. 근거 모델은 이미
# 확정된 점수를 **복사**하고 그 점수를 정당화하는 문장만 생성한다. 새 기본 경로에서는
# 제출과 같은 average_matched 정수이고, legacy human 경로만 소수를 유지한다.
#
# ## 왜 SCORE_SOURCE=human_average_matched가 기본값인가
#
# 근거 모델이 배우는 것은 "주어진 점수 X와 에세이 E에 대해, E가 왜 X인지 쓰는 법"이다. 이 능력은
# X가 무엇이든 전이된다. LLM Judge도 점수가 맞는지가 아니라 **명시된 점수와 근거가 정합적인지,
# 에세이에 근거가 실재하는지**를 본다.
#
# 인간 점수는 어느 백본이 이기든 바뀌지 않으므로 teacher A/B를 특정 score checkpoint에
# 종속시키지 않는다. 다만 소수 인간 점수를 그대로 쓰면 항상 정수인 Docker 조건과 lexical
# 분포가 갈린다. 기본값은 인간 C/O/E에 Docker와 같은 average_matched를 적용해, gold 기반
# 의미는 유지하면서 teacher prompt·student SFT가 최종 제출 정수 표면을 보게 한다.
#
# legacy 재현에는 `SCORE_SOURCE=human`을 명시한다. `SCORE_SOURCE=score_predictions`도
# 지원한다. 서빙 조건 분포에 더 가깝지만 (1) 확정 점수
# 모델이 나와야 시작할 수 있고 (2) train split에 대한 우리 예측은 학습 데이터라 실제보다
# 정확해서 낙관 편향이 있다. 시간이 남으면 그때 재생성한다.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAGE="${STAGE:?STAGE가 필요합니다: teacher|judge_server|generate|judge|train|verify|all}"
GPU="${GPU:-0}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv-train/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-${ROOT_DIR}/.venv-vllm/bin/python}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/main_code_relonation/results}"
RUN_NAME="${RUN_NAME:-report_rationale_v4}"
WORK="${OUT_ROOT}/${RUN_NAME}"

# teacher. FAQ §5가 "학습 과정에서 제출 제한보다 큰 모델을 티처로 활용"을 명시적으로 허용한다.
# 기술서의 Gemma4-26B-A4B-it AWQ 경로와 revision은 실행 시 명시한다.
TEACHER_MODEL="${TEACHER_MODEL:-}"
TEACHER_MODEL_REVISION="${TEACHER_MODEL_REVISION:-}"
if [[ "${STAGE}" == teacher || "${STAGE}" == generate ]]; then
  : "${TEACHER_MODEL:?기술서의 Gemma4-26B-A4B-it AWQ 저장소 또는 로컬 경로를 지정하세요}"
  : "${TEACHER_MODEL_REVISION:?사용할 teacher의 정확한 revision을 지정하세요}"
  : "${RATIONALE_PROMPT_FILE:?기술서 prompt_8 원문 경로가 필요합니다. v4와 동일하다고 가정하지 않습니다}"
fi
TEACHER_SERVED_MODEL="${TEACHER_SERVED_MODEL:-teacher}"
TEACHER_PORT="${TEACHER_PORT:-8100}"
TEACHER_API_BASE="${TEACHER_API_BASE:-http://127.0.0.1:${TEACHER_PORT}/v1}"
TEACHER_MAX_LEN="${TEACHER_MAX_LEN:-8192}"

# 생성 모델과 분리한 공통 proxy judge. 공식 Q4_K_M Judge와 동일 모델 계열이지만
# 양자화/실행기는 다르므로 결과는 local proxy로만 해석한다.
JUDGE_MODEL="${JUDGE_MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
JUDGE_MODEL_REVISION="${JUDGE_MODEL_REVISION:-eba8776085dddd0407a69f41c07e932e1a4d2097}"
JUDGE_SERVED_MODEL="${JUDGE_SERVED_MODEL:-proxy-judge}"
JUDGE_PORT="${JUDGE_PORT:-8300}"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:${JUDGE_PORT}/v1}"

# 학생. 점수 백본과 같은 모델을 쓴다(instruct 계열이라 chat template이 정상이다).
RECIPE="${RECIPE:-${ROOT_DIR}/main_code_relonation/recipes/r18_report_qwen35.json}"
RATIONALE_PROMPT_FILE="${RATIONALE_PROMPT_FILE:-${ROOT_DIR}/main_code_relonation/prompts/rationale_prompt_v4.txt}"
SCORE_SOURCE="${SCORE_SOURCE:-human_average_matched}"
LIMIT_ARGS=()
[[ -n "${LIMIT:-}" ]] && LIMIT_ARGS=(--limit "${LIMIT}")
# 현재 v4 student와 동일한 생성 길이 지시를 teacher에도 전달한다.
HINT_ARGS=()
TEACHER_SKELETON_HINT="${TEACHER_SKELETON_HINT:-6~9문장 450~540 tokens}"
export TEACHER_RATIONALE_CHAR_LIMIT="${TEACHER_RATIONALE_CHAR_LIMIT:-1200}"
export TEACHER_REQUIRE_QUOTES="${TEACHER_REQUIRE_QUOTES:-1}"
[[ -n "${TEACHER_SKELETON_HINT:-}" ]] && HINT_ARGS=(--skeleton-hint "${TEACHER_SKELETON_HINT}")

mkdir -p "${WORK}"

log() { printf '\n=== %s ===\n' "$*"; }
run() {
  echo "+ CUDA_VISIBLE_DEVICES=${GPU} $*"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0
  CUDA_VISIBLE_DEVICES="${GPU}" "$@"
}

# --- 입력 경로 ---------------------------------------------------------------
# prepared train은 공개 validation 400편을 제외한 11,600편
# (official_train 2,000 + origin_pool_extra 9,600)이며 prompt/essay/score가 모두 들어 있다.
TRAIN_INPUT="${TRAIN_INPUT:-${ROOT_DIR}/main_code/datasets/processed_dataset/train.jsonl}"
VALID_INPUT="${VALID_INPUT:-${ROOT_DIR}/main_code/datasets/processed_dataset/validation.jsonl}"
# 우리 점수 모델의 train split 예측. `SCORE_SOURCE=score_predictions`일 때 필요하다.
# SCORE_FILE은 기존 호출 호환용 alias이며 train 용도로만 해석한다. validation verify 점수와
# 같은 경로를 재사용하면 ID split이 섞이므로 둘을 분리한다.
TRAIN_SCORE_FILE="${TRAIN_SCORE_FILE:-${SCORE_FILE:-${WORK}/score_predictions_train.jsonl}}"
VERIFY_SCORE_FILE="${VERIFY_SCORE_FILE:-${WORK}/score_predictions_validation.jsonl}"

case "${STAGE}" in
teacher)
  log "teacher 서버 기동: ${TEACHER_MODEL} (port ${TEACHER_PORT})"
  echo "이 프로세스는 foreground로 남는다. 다른 터미널에서 generate 단계를 실행한다."
  # 이 머신에는 nvcc가 PATH에 없다(/usr/local/cuda/bin에 CUDA 13.1이 있지만 torch는 cu130
  # 빌드다). 로드 시점에 CUDA를 JIT 컴파일하는 경로 두 개가 engine core init을 통째로
  # 죽이므로 둘 다 끈다(2026-08-06 실측).
  #   VLLM_USE_DEEP_GEMM         FP8 커널 JIT -> `"NVCC compilation failed"`.
  #                              끄면 TRITON FP8 백엔드로 폴백한다(조금 느리지만 동작).
  #   VLLM_USE_FLASHINFER_SAMPLER  sampling 커널 JIT -> `No such file or directory: 'ninja'`.
  #                              어차피 temperature=0 greedy라 이 커널로 얻을 게 없다.
  export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  # MoE teacher(A3B 계열)는 위 두 스위치로 막히지 않는 **세 번째** JIT 경로를 탄다:
  # flashinfer/fused_moe/core.py -> gen_cutlass_fused_moe_sm120_module -> run_ninja.
  # ninja는 .venv-vllm에 이미 설치돼 있지만 스크립트가 venv를 활성화하지 않고 python
  # 절대경로만 쓰므로 PATH에 없어 FileNotFoundError로 engine core init이 죽었다.
  # nvcc도 함께 넣는다(컴파일에 몇 분 걸리고 첫 로드에서만 발생한다).
  export PATH="$(dirname "${VLLM_PYTHON}"):/usr/local/cuda/bin:${PATH}"
  command -v ninja >/dev/null || {
    echo "ninja를 PATH에서 찾지 못했습니다. '${VLLM_PYTHON} -m pip install ninja' 후 재시도하세요" >&2
    exit 2; }

  # MoE teacher는 bf16 가중치만으로 수십 GB다. 학습이 GPU를 쓰는 중이면 engine core가
  # "Free memory ... less than desired GPU memory utilization"으로 죽는데, 그 로그가
  # 길어 원인을 놓치기 쉽다. 먼저 확인하고 명확히 멈춘다.
  TEACHER_MIN_FREE_GIB="${TEACHER_MIN_FREE_GIB:-80}"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}")
  if (( free_mib < TEACHER_MIN_FREE_GIB * 1024 )); then
    echo "GPU ${GPU} 여유 메모리 ${free_mib} MiB < 요구 $((TEACHER_MIN_FREE_GIB * 1024)) MiB." >&2
    echo "GPU 여유 공간을 확보하거나 해당 AWQ 모델에 맞게 TEACHER_MIN_FREE_GIB/TEACHER_GPU_UTIL을 지정하세요." >&2
    exit 2
  fi
  run "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${TEACHER_MODEL}" \
    --revision "${TEACHER_MODEL_REVISION}" \
    --host 127.0.0.1 --port "${TEACHER_PORT}" \
    --max-model-len "${TEACHER_MAX_LEN}" \
    --max-num-seqs "${TEACHER_MAX_NUM_SEQS:-16}" \
    --served-model-name "${TEACHER_SERVED_MODEL}" \
    --dtype "${TEACHER_DTYPE:-auto}" \
    --generation-config vllm \
    --gpu-memory-utilization "${TEACHER_GPU_UTIL:-0.85}" \
    ${TEACHER_EXTRA_ARGS:-}
  ;;

judge_server)
  log "공통 proxy judge 서버 기동: ${JUDGE_MODEL} (port ${JUDGE_PORT})"
  export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
  export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
  # 이 proxy revision은 로컬 snapshot이 완전하지만 Hub API의 tree endpoint에서는
  # 404를 반환할 수 있다. Judge는 고정된 로컬 artifact만 사용해 원격 조회를 막는다.
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export PATH="$(dirname "${VLLM_PYTHON}"):/usr/local/cuda/bin:${PATH}"
  run "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${JUDGE_MODEL}" \
    --revision "${JUDGE_MODEL_REVISION}" \
    --host 127.0.0.1 --port "${JUDGE_PORT}" \
    --max-model-len "${JUDGE_MAX_LEN:-8192}" \
    --max-num-seqs "${JUDGE_MAX_NUM_SEQS:-16}" \
    --served-model-name "${JUDGE_SERVED_MODEL}" \
    --language-model-only \
    --reasoning-parser qwen3 \
    --dtype bfloat16 \
    --generation-config vllm \
    --gpu-memory-utilization "${JUDGE_GPU_UTIL:-0.60}"
  ;;

score_predictions)
  # 우리 점수 모델로 train split을 추론해 근거 학습 조건값을 만든다.
  : "${SCORE_CHECKPOINT:?SCORE_CHECKPOINT가 필요합니다 (예: main_code/results/.../best_checkpoint_rmse)}"
  log "점수 예측 생성: ${SCORE_CHECKPOINT} -> ${TRAIN_SCORE_FILE}"
  run "${PYTHON_BIN}" -m main_code.infer \
    --checkpoint "${SCORE_CHECKPOINT}" \
    --input "${TRAIN_INPUT}" \
    --output-dir "${WORK}/score_infer_train" \
    --batch-size 16 "${LIMIT_ARGS[@]}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    cp "${WORK}/score_infer_train/score_predictions.jsonl" "${TRAIN_SCORE_FILE}"
    echo "복사 완료: ${TRAIN_SCORE_FILE}"
  fi
  ;;

generate)
  log "teacher 근거 생성 (score_source=${SCORE_SOURCE})"
  EXTRA=()
  if [[ "${SCORE_SOURCE}" == "score_predictions" ]]; then
    [[ -f "${TRAIN_SCORE_FILE}" ]] || {
      echo "TRAIN_SCORE_FILE이 없습니다: ${TRAIN_SCORE_FILE}" >&2
      echo "  먼저: STAGE=score_predictions SCORE_CHECKPOINT=<경로> bash $0" >&2
      exit 2
    }
    EXTRA=(--score-predictions "${TRAIN_SCORE_FILE}")
  fi
  run "${PYTHON_BIN}" -m main_code_relonation.teacher_generate \
    --input "${TRAIN_INPUT}" \
    --output "${WORK}/pseudo_train.jsonl" \
    --api-base "${TEACHER_API_BASE}" \
    --model "${TEACHER_SERVED_MODEL}" \
    --model-revision "${TEACHER_MODEL}@${TEACHER_MODEL_REVISION}" \
    --chat-template vllm_served_default \
    --rationale-prompt-file "${RATIONALE_PROMPT_FILE}" \
    --score-source "${SCORE_SOURCE}" \
    --concurrency "${CONCURRENCY:-16}" \
    --max-tokens "${TEACHER_MAX_TOKENS:-2048}" \
    "${HINT_ARGS[@]}" "${EXTRA[@]}" "${LIMIT_ARGS[@]}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${PYTHON_BIN}" - "${WORK}/pseudo_train.jsonl.manifest.json" \
      "${MIN_TEACHER_SUCCESS_RATE:-0.80}" <<'PYEOF'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
minimum = float(sys.argv[2])
manifest = json.loads(path.read_text(encoding="utf-8"))
selected = int(manifest["selected"])
successes = int(manifest["generated_this_run"]) + int(manifest["resumed_skips"])
rate = successes / selected if selected else 0.0
print(f"teacher valid rows: {successes}/{selected} ({rate:.2%})")
if selected < 1 or rate < minimum:
    raise SystemExit(f"teacher success rate {rate:.2%} < required {minimum:.2%}")
PYEOF
  fi
  ;;

judge)
  log "독립 공통 proxy judge로 형식/정합성 실패 제거"
  run "${PYTHON_BIN}" -m main_code_relonation.teacher_judge \
    --pseudo "${WORK}/pseudo_train.jsonl" \
    --output "${WORK}/judge_results.jsonl" \
    --accepted-output "${WORK}/accepted_pseudo_train.jsonl" \
    --api-base "${JUDGE_API_BASE}" \
    --model "${JUDGE_SERVED_MODEL}" \
    --model-revision "${JUDGE_MODEL}@${JUDGE_MODEL_REVISION}" \
    --concurrency "${CONCURRENCY:-16}" \
    "${LIMIT_ARGS[@]}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${PYTHON_BIN}" - "${WORK}/judge_results.jsonl.manifest.json" \
      "${MIN_JUDGE_PARSE_RATE:-0.90}" <<'PYEOF'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
minimum = float(sys.argv[2])
manifest = json.loads(path.read_text(encoding="utf-8"))
selected = int(manifest["selected"])
valid = int(manifest["valid_latest"])
rate = valid / selected if selected else 0.0
print(f"proxy Judge parseable rows: {valid}/{selected} ({rate:.2%})")
if selected < 1 or rate < minimum:
    raise SystemExit(f"proxy Judge parse rate {rate:.2%} < required {minimum:.2%}")
PYEOF
  fi
  ;;

train)
  log "새 LoRA 학습 (score_mode=fixed, 점수 재생성 없음)"
  PSEUDO="${PSEUDO_FILE:-${WORK}/pseudo_train.jsonl}"
  [[ -f "${PSEUDO}" || "${DRY_RUN:-0}" == "1" ]] || {
    echo "학습 입력이 없습니다: ${PSEUDO}" >&2; exit 2; }
  run "${PYTHON_BIN}" -m main_code_relonation.train \
    --recipe "${RECIPE}" \
    --rationale-prompt-file "${RATIONALE_PROMPT_FILE}" \
    --train-file "${PSEUDO}" \
    --output-dir "${WORK}/lora" \
    "${LIMIT_ARGS[@]}"
  ;;

verify)
  log "학습 어댑터로 추론 검증 (형식 준수 + 점수 보존)"
  # `score_mode=fixed`는 조건이 될 점수 파일을 요구한다. 별도 파일이 없으면 validation
  # 인간 점수로 만들되, 현 기본 경로는 학습과 같은 average_matched 정수로 변환한다.
  if [[ ! -f "${VERIFY_SCORE_FILE}" && "${DRY_RUN:-0}" != "1" ]]; then
    echo "VERIFY_SCORE_FILE이 없어 validation 인간 점수로 만든다: ${VERIFY_SCORE_FILE}"
    "${PYTHON_BIN}" - "${VALID_INPUT}" "${VERIFY_SCORE_FILE}" "${SCORE_SOURCE}" <<'PYEOF'
import json, sys, pathlib
from main_code.postprocess import ScorePostprocessor

source, target, score_source = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
target.parent.mkdir(parents=True, exist_ok=True)
postprocessor = ScorePostprocessor("average_matched")
written = 0
with target.open("w", encoding="utf-8") as stream:
    # JSONL은 split("\n")으로 읽는다. str.splitlines()는 essay 안의 U+2028 같은 Unicode
    # 줄바꿈에서도 쪼개져 행이 깨진다(train.jsonl에 실제로 존재한다).
    for line in source.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("score") or {}
        traits = ("content", "organization", "expression")
        if not all(t in score for t in traits):
            continue
        fixed = {t: float(score[t]) for t in traits}
        if score_source == "human_average_matched":
            fixed = postprocessor.apply_row(fixed)
        stream.write(json.dumps(
            {"essay_id": str(row["id"]), "scores": fixed},
            ensure_ascii=False) + "\n")
        written += 1
print(f"  기록 {written}행")
PYEOF
  fi
  run "${PYTHON_BIN}" -m main_code_relonation.infer \
    --recipe "${RECIPE}" \
    --input "${VALID_INPUT}" \
    --score-predictions "${VERIFY_SCORE_FILE}" \
    --adapter-path "${WORK}/lora/final_adapter" \
    --rationale-prompt-file "${RATIONALE_PROMPT_FILE}" \
    --output-dir "${WORK}/verify" \
    "${LIMIT_ARGS[@]}"
  echo
  echo "manifest에 연결할 값:"
  echo "  adapter               ${WORK}/lora/final_adapter"
  echo "  chat_template_sha256  \$(jq -r .chat_template_hash ${WORK}/lora/tokenization_audit.json)"
  echo "  base_model            \$(jq -r .model_id ${WORK}/lora/resolved_config.json)"
  echo "  max_length            \$(jq -r .max_length ${WORK}/lora/resolved_config.json)"
  ;;

all)
  echo "generate와 judge는 서로 다른 서버를 사용하므로 STAGE=all은 지원하지 않습니다." >&2
  echo "단일 단계 또는 run_rationale_teacher_ab.sh를 사용하십시오." >&2
  exit 2
  ;;

*)
  echo "STAGE는 teacher|judge_server|score_predictions|generate|judge|train|verify|all 중 하나여야 합니다" >&2
  exit 2
  ;;
esac

echo
echo "작업 폴더: ${WORK}"
