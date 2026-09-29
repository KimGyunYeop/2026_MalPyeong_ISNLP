#!/usr/bin/env bash
# 확정된 final manifest로 image를 build하고 같은 image ID의 400편 exact 검증 뒤 push한다.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

[[ "$#" == "1" ]] || {
  echo "사용법: GPU=all PUSH=0|1 REGISTRY_REPO=OWNER/REPO bash main_code_submission/release_final.sh MANIFEST" >&2
  exit 2
}

MANIFEST="$(python3 - "$1" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]).expanduser().resolve()
if not path.is_file():
    raise SystemExit(f"manifest가 없습니다: {path}")
print(path)
PY
)"

cd "${ROOT}"

LOCAL_REPO="malpyeong-writing-scorer"
REGISTRY_REPO="${REGISTRY_REPO:-gyunyeop/writing-scorer}"
GPU="${GPU:-all}"
PUSH="${PUSH:-0}"
RESULTS="${RESULTS:-main_code_submission/results/release_final}"

[[ "${PUSH}" == "0" || "${PUSH}" == "1" ]] || {
  echo "PUSH는 0 또는 1이어야 합니다" >&2
  exit 2
}
[[ "${REGISTRY_REPO}" == */* ]] || {
  echo "REGISTRY_REPO는 OWNER/REPO 형식이어야 합니다" >&2
  exit 2
}

mapfile -t RELEASE_FIELDS < <(python3 - "${MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
extra = manifest.get("extra")
if not isinstance(extra, dict):
    raise SystemExit("manifest.extra는 JSON object여야 합니다")

tag = extra.get("expected_image_tag")
prediction = extra.get("expected_http_prediction_sha256")
if not isinstance(tag, str) or re.fullmatch(
    r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag
) is None:
    raise SystemExit("manifest.extra.expected_image_tag가 유효한 고정 Docker tag가 아닙니다")
if not isinstance(prediction, str) or re.fullmatch(r"[0-9a-f]{64}", prediction) is None:
    raise SystemExit(
        "manifest.extra.expected_http_prediction_sha256는 64자리 소문자 SHA-256이어야 합니다"
    )

print(tag)
print(prediction)
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)
[[ "${#RELEASE_FIELDS[@]}" == "3" ]] || {
  echo "manifest release field를 읽지 못했습니다: ${MANIFEST}" >&2
  exit 2
}
TAG="${RELEASE_FIELDS[0]}"
EXPECTED_PREDICTION_SHA256="${RELEASE_FIELDS[1]}"
MANIFEST_SHA256="${RELEASE_FIELDS[2]}"

command -v docker >/dev/null || { echo "docker가 없습니다" >&2; exit 1; }
DOCKER_VIA_SG=0
if ! docker info >/dev/null 2>&1; then
  if command -v sg >/dev/null && sg docker -c 'docker info >/dev/null 2>&1'; then
    DOCKER_VIA_SG=1
    echo "현재 셸에 docker 그룹이 미반영되어 sg docker -c를 사용합니다"
  else
    echo "Docker daemon에 접근할 수 없습니다. 재로그인 또는 newgrp docker를 실행하세요" >&2
    exit 1
  fi
fi
_docker() {
  if [[ "${DOCKER_VIA_SG}" == "0" ]]; then
    docker "$@"
  else
    local command_line
    printf -v command_line '%q ' docker "$@"
    sg docker -c "${command_line}"
  fi
}

LOCAL_IMAGE="${LOCAL_REPO}:${TAG}"
REMOTE_IMAGE="${REGISTRY_REPO}:${TAG}"
RUN_DIR="${RESULTS}/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

# 검증한 image와 평가기가 pull할 image가 갈리지 않도록 원격 tag 재사용을 막는다.
if [[ "${PUSH}" == "1" ]] && _docker manifest inspect "${REMOTE_IMAGE}" >/dev/null 2>&1; then
  echo "원격 tag가 이미 존재합니다. manifest에 새 고정 tag를 선언하세요: ${REMOTE_IMAGE}" >&2
  exit 1
fi

echo "[1/3] build ${LOCAL_IMAGE}"
IMAGE_NAME="${LOCAL_REPO}" \
  bash main_code_submission/build_image.sh "${MANIFEST}" "${TAG}" \
  2>&1 | tee "${RUN_DIR}/build.log"

IMAGE_ID="$(_docker image inspect "${LOCAL_IMAGE}" --format '{{.Id}}')"
echo "[2/3] 같은 image ID로 no-argument API + validation 400편 exact 검사"
EXPECTED_PREDICTION_SHA256="${EXPECTED_PREDICTION_SHA256}" \
  PULL=0 GPU="${GPU}" LIMIT= RESULT_ROOT="${RUN_DIR}/check" \
  bash code_for_docker_check_otherserv/run_docker_check.sh "${IMAGE_ID}" \
  2>&1 | tee "${RUN_DIR}/check.log"

write_metadata() {
  local digest="${1:-}"
  {
    echo "schema_version=1"
    echo "manifest=${MANIFEST}"
    echo "manifest_sha256=${MANIFEST_SHA256}"
    echo "prediction_sha256=${EXPECTED_PREDICTION_SHA256}"
    echo "image=${REMOTE_IMAGE}"
    echo "image_id=${IMAGE_ID}"
    if [[ -n "${digest}" ]]; then
      echo "digest=${REGISTRY_REPO}@${digest}"
      echo "submission_url=docker://${REMOTE_IMAGE}"
    fi
    echo "results=${RUN_DIR}"
  } | tee "${RUN_DIR}/release.txt"
}

if [[ "${PUSH}" == "0" ]]; then
  echo "[3/3] push 생략"
  write_metadata
  exit 0
fi

echo "[3/3] push ${REMOTE_IMAGE}"
_docker tag "${IMAGE_ID}" "${REMOTE_IMAGE}"
_docker push "${REMOTE_IMAGE}" 2>&1 | tee "${RUN_DIR}/push.log"
DIGEST="$(sed -nE 's/.*digest: (sha256:[0-9a-f]{64}).*/\1/p' \
  "${RUN_DIR}/push.log" | tail -1)"
[[ "${DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
  echo "push 뒤 RepoDigest를 확인하지 못했습니다" >&2
  exit 1
}
REPO_DIGEST="${REGISTRY_REPO}@${DIGEST}"
write_metadata "${DIGEST}"

echo "제출 URL: docker://${REMOTE_IMAGE}"
echo "타 서버 검사용 metadata: ${RUN_DIR}/release.txt"
echo "타 서버 검사용 digest: ${REPO_DIGEST}"
