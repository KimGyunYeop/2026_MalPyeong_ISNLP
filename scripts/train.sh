#!/usr/bin/env bash
# 학습.
#
#   GPU=0 bash scripts/train.sh score
#       채점 LoRA를 시드 42~49로 순차 학습한다(report_abcd.json).
#       이미 끝난 시드(run.json과 checkpoint가 있는 출력)는 건너뛴다.
#   GPU=0 bash scripts/train.sh rationale [PSEUDO_JSONL]
#       근거 LoRA를 학습한다(r18_report_qwen35.json).
#       기본 입력은 scripts/prepare_data.sh generate의 출력이다.
#
# 결과: ${OUT_ROOT}/score_s42 ~ score_s49, ${OUT_ROOT}/rationale
# 환경변수: PYTHON_BIN, GPU, DATASET_ROOT, OUT_ROOT, SEEDS
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
GPU="${GPU:-0}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/main_code/datasets}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/report_abcd}"
SCORE_CONFIG="main_code/configs/report_abcd.json"
RATIONALE_RECIPE="main_code_relonation/recipes/r18_report_qwen35.json"

case "${1:-}" in
score)
  for seed in ${SEEDS:-42 43 44 45 46 47 48 49}; do
    out="${OUT_ROOT}/score_s${seed}"
    if [[ -f "${out}/run.json" && -d "${out}/checkpoint" ]]; then
      echo "[skip] seed ${seed}: ${out}/checkpoint"
      continue
    fi
    echo "[score] seed ${seed} -> ${out}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m main_code.train \
      --config "${SCORE_CONFIG}" \
      --dataset-root "${DATASET_ROOT}" \
      --seed "${seed}" \
      --output-dir "${out}"
  done
  ;;
rationale)
  PSEUDO="${2:-${ROOT}/main_code_relonation/results/report_rationale_v4/pseudo_train.jsonl}"
  [[ -f "${PSEUDO}" ]] || {
    echo "근거 학습 데이터가 없습니다: ${PSEUDO}" >&2
    echo "먼저 bash scripts/prepare_data.sh generate MODEL REVISION을 실행하세요" >&2
    exit 2
  }
  out="${OUT_ROOT}/rationale"
  if [[ -f "${out}/completed.json" ]]; then
    echo "[skip] rationale: ${out}/final_adapter"
    exit 0
  fi
  echo "[rationale] ${PSEUDO} -> ${out}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m main_code_relonation.train \
    --recipe "${RATIONALE_RECIPE}" \
    --train-file "${PSEUDO}" \
    --output-dir "${out}"
  ;;
*)
  echo "사용법: GPU=0 bash scripts/train.sh score | rationale [PSEUDO_JSONL]" >&2
  exit 2
  ;;
esac
