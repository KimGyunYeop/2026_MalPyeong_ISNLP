#!/usr/bin/env bash
# 다른 GPU 서버에서 final release digest를 anonymous pull하고 full 400편 exact 검사한다.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

[[ "$#" -ge 1 && "$#" -le 2 ]] || {
  echo "사용법:" >&2
  echo "  bash check_uploaded_final.sh /PATH/TO/release.txt" >&2
  echo "  bash check_uploaded_final.sh OWNER/REPO@sha256:<64hex> <prediction-sha256>" >&2
  exit 2
}

INPUT="$1"
CALLER_EXPECTED="${EXPECTED_PREDICTION_SHA256:-}"
if [[ -f "${INPUT}" ]]; then
  [[ "$#" == "1" ]] || {
    echo "release metadata를 사용할 때 prediction hash 인자를 따로 넣지 마세요" >&2
    exit 2
  }
  mapfile -t RELEASE_FIELDS < <(python3 - "${INPUT}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
fields: dict[str, str] = {}
for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not raw.strip():
        continue
    if "=" not in raw:
        raise SystemExit(f"{path}:{line_number}: key=value 형식이 아닙니다")
    key, value = raw.split("=", 1)
    if key in fields:
        raise SystemExit(f"{path}:{line_number}: 중복 key입니다: {key}")
    fields[key] = value

if fields.get("schema_version") != "1":
    raise SystemExit(f"{path}: schema_version=1 metadata가 아닙니다")
for key in ("digest", "prediction_sha256"):
    if not fields.get(key):
        raise SystemExit(f"{path}: {key}가 없습니다. PUSH=1 release metadata가 필요합니다")
print(fields["digest"])
print(fields["prediction_sha256"])
PY
  )
  [[ "${#RELEASE_FIELDS[@]}" == "2" ]] || {
    echo "release metadata를 읽지 못했습니다: ${INPUT}" >&2
    exit 2
  }
  IMAGE_REF="${RELEASE_FIELDS[0]}"
  EXPECTED_PREDICTION_SHA256="${RELEASE_FIELDS[1]}"
  if [[ -n "${CALLER_EXPECTED}" && "${CALLER_EXPECTED}" != "${EXPECTED_PREDICTION_SHA256}" ]]; then
    echo "환경변수 prediction hash와 release metadata가 다릅니다" >&2
    exit 2
  fi
else
  IMAGE_REF="${INPUT}"
  ARG_EXPECTED="${2:-}"
  if [[ -n "${CALLER_EXPECTED}" && -n "${ARG_EXPECTED}" \
      && "${CALLER_EXPECTED}" != "${ARG_EXPECTED}" ]]; then
    echo "인자와 환경변수의 prediction hash가 다릅니다" >&2
    exit 2
  fi
  EXPECTED_PREDICTION_SHA256="${ARG_EXPECTED:-${CALLER_EXPECTED}}"
fi

IMAGE_REF="${IMAGE_REF#docker://}"
[[ "${IMAGE_REF}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]] || {
  echo "공개 registry digest가 필요합니다: OWNER/REPO@sha256:<64hex>" >&2
  exit 2
}
[[ "${EXPECTED_PREDICTION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "확정된 64자리 소문자 prediction SHA-256이 필요합니다" >&2
  exit 2
}

cd "${HERE}"
[[ "${GPU:-all}" == "all" ]] || {
  echo "최종 타 서버 검사는 평가 형태와 같은 GPU=all만 허용합니다" >&2
  exit 2
}
GPU="all"
EMPTY_DOCKER_CONFIG="$(mktemp -d)"
cleanup() { rm -rf -- "${EMPTY_DOCKER_CONFIG}"; }
trap cleanup EXIT
unset DOCKER_AUTH_CONFIG REGISTRY_AUTH_FILE
export DOCKER_CONFIG="${EMPTY_DOCKER_CONFIG}"

RUN_ID="uploaded_final_$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${PWD}/results/${RUN_ID}"

EXPECTED_PREDICTION_SHA256="${EXPECTED_PREDICTION_SHA256}" \
  PULL=1 GPU="${GPU}" LIMIT= RESULT_ROOT="${RESULT_ROOT}" \
  bash run_docker_check.sh "${IMAGE_REF}"

python3 - "${RESULT_ROOT}/evaluation/summary.json" \
  "${RESULT_ROOT}/evaluation/metrics.json" \
  "${RESULT_ROOT}/evaluation/prediction_hash.json" \
  "${EXPECTED_PREDICTION_SHA256}" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("count") != 400 or summary.get("all_gates_passed") is not True:
    raise SystemExit(f"검증 실패: {summary}")

metrics = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
prediction = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
expected = sys.argv[4]
if prediction != {
    "format": "essay_id\\tcontent_int\\torganization_int\\texpression_int\\n",
    "count": 400,
    "expected_sha256": expected,
    "actual_sha256": expected,
    "match": True,
}:
    raise SystemExit(f"prediction exact 검증 실패: {prediction}")

official = metrics["official"]["official_after_per_trait_half_up"]
raw = metrics["official"]["raw_continuous"]
print("PASS: anonymous digest pull + no-argument GPU server + validation 400/400")
print(f"prediction SHA-256={expected}")
print(f"official RMSE={official['rmse']} Spearman={official['spearman']}")
print(f"HTTP raw RMSE={raw['rmse']} Spearman={raw['spearman']}")
PY

echo "결과 폴더: ${RESULT_ROOT}"
