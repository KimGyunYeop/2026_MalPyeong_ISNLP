#!/usr/bin/env bash
# Docker 제출.
#
#   bash scripts/submit.sh build TAG
#       학습 산출물을 검사해 manifest를 만들고 이미지를 빌드한다. 컨테이너로 공식
#       validation 400편을 채점해 정수 점수 해시를 기록한다.
#   bash scripts/submit.sh push OWNER/REPO TAG
#       기록한 해시로 docker_release.sh push를 실행한다. 이미지를 다시 빌드하고 같은
#       400편 결과를 재검증한 뒤 업로드한다. 먼저 docker login을 실행한다.
#
# 공식 validation 원본은 main_code/datasets/raw_dataset/official_competition/에서 찾고,
# 다른 위치에 있으면 DATA_FILE로 지정한다.
# 환경변수: PYTHON_BIN, GPU(기본 all), OUT_ROOT, DATA_FILE, PORT
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
export PYTHON_BIN
export OUT_ROOT="${OUT_ROOT:-${ROOT}/results/report_abcd}"
export MANIFEST_DIR="${MANIFEST_DIR:-${ROOT}/main_code_submission/results/report_manifests}"
GPU="${GPU:-all}"
LOCAL_REPO="malpyeong-writing-scorer"
SERVED_MODEL="malpyeong-writing-scorer"

usage() {
  echo "사용법: bash scripts/submit.sh build TAG | push OWNER/REPO TAG" >&2
  exit 2
}

find_data_file() {
  if [[ -z "${DATA_FILE:-}" ]]; then
    local found=()
    mapfile -t found < <(find "${ROOT}/main_code/datasets/raw_dataset/official_competition" \
      -maxdepth 1 -name '*validation*.jsonl' 2>/dev/null | sort)
    [[ "${#found[@]}" == "1" ]] || {
      echo "공식 validation 원본을 하나로 찾지 못했습니다. DATA_FILE로 지정하세요" >&2
      exit 2
    }
    DATA_FILE="${found[0]}"
  fi
  [[ -f "${DATA_FILE}" ]] || { echo "DATA_FILE이 없습니다: ${DATA_FILE}" >&2; exit 2; }
  DATA_FILE="$(cd -- "$(dirname -- "${DATA_FILE}")" && pwd)/$(basename -- "${DATA_FILE}")"
  export DATA_FILE
}

hash_file() {
  echo "${MANIFEST_DIR}/${1}.prediction_sha256"
}

case "${1:-}" in
build)
  [[ "$#" == "2" ]] || usage
  TAG="$2"
  find_data_file

  echo "[1/3] 학습 산출물 검사와 manifest 생성"
  IMAGE_TAG="${TAG}" EXPECTED_HTTP_PREDICTION_SHA256="" bash docker_release.sh check
  MANIFEST="${MANIFEST_DIR}/${TAG}_.json"

  echo "[2/3] 이미지 빌드: ${LOCAL_REPO}:${TAG}"
  IMAGE_NAME="${LOCAL_REPO}" bash main_code_submission/build_image.sh "${MANIFEST}" "${TAG}"

  echo "[3/3] 공식 validation 400편 채점"
  RUN_DIR="${ROOT}/main_code_submission/results/submit_build/${TAG}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${RUN_DIR}"
  if [[ "${GPU}" == "all" ]]; then GPU_SPEC="all"; else GPU_SPEC="device=${GPU}"; fi
  PORT="${PORT:-8000}"
  CONTAINER="malpyeong-submit-build-$$"
  docker run --rm -d --name "${CONTAINER}" --gpus "${GPU_SPEC}" \
    -p "127.0.0.1:${PORT}:8000" "${LOCAL_REPO}:${TAG}" >/dev/null
  trap 'docker logs "${CONTAINER}" > "${RUN_DIR}/container.log" 2>&1 || true;
        docker stop "${CONTAINER}" >/dev/null 2>&1 || true' EXIT
  python3 code_for_docker_check_otherserv/evaluate_http.py \
    --base-url "http://127.0.0.1:${PORT}" \
    --input "${DATA_FILE}" \
    --output-dir "${RUN_DIR}/evaluation" \
    --expected-count 400 \
    --expected-model "${SERVED_MODEL}"

  # run_docker_check.sh가 비교하는 것과 같은 essay_id/C/O/E 정수열 해시다.
  PREDICTION_SHA256="$(python3 - "${RUN_DIR}/evaluation" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if summary.get("count") != 400 or not summary.get("all_gates_passed"):
    raise SystemExit(f"validation 검사 실패: {root / 'summary.json'}")
rows = [json.loads(line) for line in
        (root / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
if len(rows) != 400:
    raise SystemExit(f"records가 400편이 아닙니다: {len(rows)}")
canonical = "".join(
    f"{r['essay_id']}\t{r['official_scores']['content']}\t"
    f"{r['official_scores']['organization']}\t{r['official_scores']['expression']}\n"
    for r in rows
)
print(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
PY
)"
  echo "${PREDICTION_SHA256}" > "$(hash_file "${TAG}")"
  echo "PASS: ${TAG} validation 400편, prediction_sha256=${PREDICTION_SHA256}"
  echo "결과: ${RUN_DIR}"
  echo "다음: docker login 후 bash scripts/submit.sh push OWNER/REPO ${TAG}"
  ;;
push)
  [[ "$#" == "3" ]] || usage
  REPO="$2"
  TAG="$3"
  HASH_FILE="$(hash_file "${TAG}")"
  [[ -f "${HASH_FILE}" ]] || {
    echo "예측 해시가 없습니다: ${HASH_FILE}" >&2
    echo "먼저 bash scripts/submit.sh build ${TAG}를 실행하세요" >&2
    exit 2
  }
  find_data_file
  REGISTRY_REPO="${REPO}" IMAGE_TAG="${TAG}" GPU="${GPU}" \
    EXPECTED_HTTP_PREDICTION_SHA256="$(cat "${HASH_FILE}")" \
    bash docker_release.sh push
  RELEASE="$(ls -td "${ROOT}"/main_code_submission/results/release_final/"${TAG}"_*/ | head -1)release.txt"
  echo
  echo "제출 정보: ${RELEASE}"
  grep '^submission_url=' "${RELEASE}"
  ;;
*)
  usage
  ;;
esac
