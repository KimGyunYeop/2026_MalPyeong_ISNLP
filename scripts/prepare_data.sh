#!/usr/bin/env bash
# 데이터 준비.
#
#   bash scripts/prepare_data.sh
#       공식 자료와 원천 자료로 학습 11,600편·검증 400편을 만든다.
#   VLLM_PYTHON=/path/to/vllm/bin/python bash scripts/prepare_data.sh teacher MODEL REVISION
#       teacher vLLM 서버를 띄운다(foreground). 다른 터미널에서 generate를 실행한다.
#   bash scripts/prepare_data.sh generate MODEL REVISION
#       사람 점수에 SMR을 적용한 정수 점수를 조건으로 근거를 생성한다.
#       파싱·점수 복사·QC에 실패한 행은 학습 단계에서 제외된다. proxy judge는 쓰지 않는다.
#       성공률은 출력만 하고 막지 않는다. 하한이 필요하면 MIN_TEACHER_SUCCESS_RATE를 지정한다.
#
# 환경변수: PYTHON_BIN, GPU, DATASET_ROOT, FORCE=1(기존 출력 덮어쓰기),
#           RATIONALE_PROMPT_FILE, TEACHER_OUT_ROOT, RUN_NAME, MIN_TEACHER_SUCCESS_RATE
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT}/.venv-train/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv-train/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/main_code/datasets}"
ACTION="${1:-data}"

usage() {
  echo "사용법: bash scripts/prepare_data.sh [data | teacher MODEL REVISION | generate MODEL REVISION]" >&2
  exit 2
}

case "${ACTION}" in
data)
  RAW_ROOT="${DATASET_ROOT}/raw_dataset"
  FORCE_ARGS=()
  [[ "${FORCE:-0}" == "1" ]] && FORCE_ARGS=(--force)
  "${PYTHON_BIN}" -m main_code.prepare_data \
    --raw-root "${RAW_ROOT}" \
    --official-root "${RAW_ROOT}/official_competition" \
    --output-root "${DATASET_ROOT}" \
    --datasets competition \
    "${FORCE_ARGS[@]}"
  for split in train validation; do
    echo "${split}: $(wc -l < "${DATASET_ROOT}/processed_dataset/${split}.jsonl")편"
  done
  ;;
teacher|generate)
  [[ "$#" == "3" ]] || usage
  # 근거 학습 recipe(r18)와 같은 v4 prompt, 사람 점수 SMR 정수 조건을 사용한다.
  STAGE="${ACTION}" \
    TEACHER_MODEL="$2" \
    TEACHER_MODEL_REVISION="$3" \
    RATIONALE_PROMPT_FILE="${RATIONALE_PROMPT_FILE:-${ROOT}/main_code_relonation/prompts/rationale_prompt_v4.txt}" \
    OUT_ROOT="${TEACHER_OUT_ROOT:-${ROOT}/main_code_relonation/results}" \
    RUN_NAME="${RUN_NAME:-report_rationale_v4}" \
    TRAIN_INPUT="${DATASET_ROOT}/processed_dataset/train.jsonl" \
    SCORE_SOURCE=human_average_matched \
    MIN_TEACHER_SUCCESS_RATE="${MIN_TEACHER_SUCCESS_RATE:-0}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash main_code_relonation/run_rationale_pipeline.sh
  ;;
*)
  usage
  ;;
esac
