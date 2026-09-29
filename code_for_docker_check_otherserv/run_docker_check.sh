#!/usr/bin/env bash
# Local image ID 또는 pull한 registry digest를 제출 형태 그대로 실행해 검사한다.
set -euo pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_REF="${1:-${IMAGE_REF:-}}"
PULL="${PULL:-0}"
GPU="${GPU:-all}"
PORT="${PORT:-8000}"
LIMIT="${LIMIT:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
# 규정 §11-4. 현행 이미지 실측이 약 39.5 GiB이므로 회귀만 잡는 느슨한 상한이다.
IMAGE_MAX_GIB="${IMAGE_MAX_GIB:-60}"
PULL_WARN_SECONDS="${PULL_WARN_SECONDS:-1800}"
RESULT_ROOT="${RESULT_ROOT:-${HERE}/results/check_$(date +%Y%m%d_%H%M%S)_$$}"
DATA_FILE="${DATA_FILE:-${HERE}/data/official_validation_400.jsonl}"
# 직접 호출하는 기존 Y6 검사는 역사적 oracle을 그대로 쓴다. 새 final release는
# 확정된 manifest의 HTTP 400편 oracle을 이 환경변수로 반드시 덮어쓴다.
EXPECTED_PREDICTION_SHA256="${EXPECTED_PREDICTION_SHA256:-47e2aad8cdb64311759e151fa7f12f1e63e5f95d48117a4fdafd50426fe14c72}"

[[ -n "${IMAGE_REF}" ]] || {
  echo "사용법: PULL=0|1 GPU=all|0 bash run_docker_check.sh IMAGE" >&2
  exit 2
}
IMAGE_REF="${IMAGE_REF#docker://}"
[[ "${PULL}" == "0" || "${PULL}" == "1" ]] || {
  echo "PULL은 0 또는 1이어야 합니다" >&2
  exit 2
}
[[ "${EXPECTED_PREDICTION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "EXPECTED_PREDICTION_SHA256는 64자리 소문자 SHA-256이어야 합니다" >&2
  exit 2
}
[[ -f "${DATA_FILE}" ]] || {
  echo "validation data가 없습니다: ${DATA_FILE}" >&2
  exit 2
}
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

mkdir -p "${RESULT_ROOT}"
if [[ "${PULL}" == "1" ]]; then
  # 규정 §11-4: "이미지가 너무 커서 docker pull 시간이 과도하게 길어짐".
  # 평가기는 우리 이미지를 받아야 시작할 수 있으므로 pull 소요를 실측해 남긴다.
  PULL_STARTED=${SECONDS}
  _docker pull "${IMAGE_REF}" 2>&1 | tee "${RESULT_ROOT}/pull.log"
  PULL_SECONDS=$((SECONDS - PULL_STARTED))
  echo "docker pull 소요: ${PULL_SECONDS}초"
  if (( PULL_SECONDS > PULL_WARN_SECONDS )); then
    echo "경고: pull이 ${PULL_SECONDS}초 걸렸습니다(기준 ${PULL_WARN_SECONDS}초). 규정 §11-4 위험" >&2
  fi
else
  PULL_SECONDS=-1
fi

IMAGE_ID="$(_docker image inspect "${IMAGE_REF}" --format '{{.Id}}')"

# 규정 §11-4 이미지 크기 게이트. 실측 크기를 기록하고 상한을 넘으면 중단한다.
IMAGE_BYTES="$(_docker image inspect "${IMAGE_REF}" --format '{{.Size}}')"
printf '{"image_bytes":%s,"image_gib":%.2f,"limit_gib":%s,"pull_seconds":%s}\n' \
  "${IMAGE_BYTES}" "$(awk -v b="${IMAGE_BYTES}" 'BEGIN{printf "%.2f", b/1073741824}')" \
  "${IMAGE_MAX_GIB}" "${PULL_SECONDS}" > "${RESULT_ROOT}/image_size.json"
awk -v b="${IMAGE_BYTES}" -v lim="${IMAGE_MAX_GIB}" 'BEGIN{
  g=b/1073741824;
  printf "이미지 크기: %.2f GiB (상한 %s GiB)\n", g, lim;
  if (g > lim) { printf "이미지가 상한을 초과했습니다. 규정 §11-4\n" > "/dev/stderr"; exit 1 }
}'
_docker image inspect "${IMAGE_REF}" > "${RESULT_ROOT}/image_inspect.json"
printf '{"image_ref":"%s","image_id":"%s"}\n' \
  "${IMAGE_REF}" "${IMAGE_ID}" > "${RESULT_ROOT}/image_identity.json"

# 평가망에서 HOME/default user가 달라도 baked base model만으로 시작할 수 있어야 한다.
# host cache를 mount하지 않고, network를 완전히 끈 uid 1000 컨테이너에서 image가 선언한
# 고정 HF cache로 config/tokenizer/weight shard 전부를 먼저 해석한다. 이 probe가 실패하면
# GPU 서버를 띄우지 않는다.
_docker run --rm -i \
  --network none \
  --user 1000:1000 \
  --read-only \
  --tmpfs "/tmp:rw,nosuid,nodev,size=256m" \
  -e HOME=/tmp \
  --entrypoint python \
  "${IMAGE_ID}" - 2>&1 <<'PY_OFFLINE_ASSET_PROBE' | tee "${RESULT_ROOT}/offline_asset_probe.log"
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoConfig, AutoTokenizer


EXPECTED_HUB = Path("/opt/submission/hf/hub").resolve()
MANIFEST = Path("/opt/submission/submission_manifest.json")

if os.geteuid() != 1000:
    raise SystemExit(f"offline asset probe가 uid 1000이 아닙니다: {os.geteuid()}")
if os.environ.get("HOME") != "/tmp":
    raise SystemExit(f"offline asset probe HOME이 /tmp가 아닙니다: {os.environ.get('HOME')!r}")
if (
    os.environ.get("HF_HUB_OFFLINE") != "1"
    or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
):
    raise SystemExit("image가 HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1을 선언해야 합니다")
if Path(HF_HUB_CACHE).resolve() != EXPECTED_HUB:
    raise SystemExit(
        "image의 HF_HUB_CACHE가 고정 cache를 가리키지 않습니다: "
        f"{HF_HUB_CACHE!r} != {str(EXPECTED_HUB)!r}"
    )
if not EXPECTED_HUB.is_dir():
    raise SystemExit(f"baked HF cache가 없습니다: {EXPECTED_HUB}")
if not MANIFEST.is_file():
    raise SystemExit(f"submission manifest가 없습니다: {MANIFEST}")


def require_bundled(path: str, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    if resolved != EXPECTED_HUB and EXPECTED_HUB not in resolved.parents:
        raise SystemExit(f"{label}가 baked cache 밖에서 해석됐습니다: {resolved}")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SystemExit(f"{label}가 비었거나 파일이 아닙니다: {resolved}")
    return resolved


payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
root = Path(payload.get("root") or "/opt/submission")
models: dict[tuple[str, str], bool] = {}
for member in payload.get("score_members") or []:
    checkpoint = Path(member["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    key = (str(config["model_id"]), str(config.get("model_revision") or "main"))
    models[key] = bool(config.get("trust_remote_code", True))

rationale = payload.get("rationale") or {}
if rationale.get("enabled", True) and rationale.get("base_model"):
    key = (
        str(rationale["base_model"]),
        str(rationale.get("base_model_revision") or "main"),
    )
    models.setdefault(key, True)
if not models:
    raise SystemExit("offline probe 대상 base model이 manifest에 없습니다")

for (repo_id, revision), trust_remote_code in sorted(models.items()):
    config = AutoConfig.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    require_bundled(
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            local_files_only=True,
        ),
        f"{repo_id}@{revision} config",
    )

    index_path = require_bundled(
        hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            revision=revision,
            local_files_only=True,
        ),
        f"{repo_id}@{revision} weight index",
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set((index.get("weight_map") or {}).values()))
    if not shards:
        raise SystemExit(f"weight index에 shard가 없습니다: {index_path}")
    for shard in shards:
        require_bundled(
            hf_hub_download(
                repo_id=repo_id,
                filename=shard,
                revision=revision,
                local_files_only=True,
            ),
            f"{repo_id}@{revision} weight shard {shard}",
        )

    commit = getattr(config, "_commit_hash", None) or revision
    print(
        f"PASS offline assets: {repo_id}@{revision} -> {commit}; "
        f"weight_shards={len(shards)}; hub={EXPECTED_HUB}"
    )
PY_OFFLINE_ASSET_PROBE

CONTAINER="malpyeong-final-check-$$"
# 규정 §1~2는 평가기가 `--network eval-net`으로 컨테이너를 띄우고 **컨테이너 이름**으로
# 접근한다고 명시한다. host port publish만으로는 그 경로를 그대로 재현하지 못하므로
# 같은 이름의 전용 bridge network를 만들어 아래에서 두 방식 모두 확인한다.
EVAL_NET="malpyeong-eval-net-$$"
cleanup() {
  _docker logs "${CONTAINER}" > "${RESULT_ROOT}/container.log" 2>&1 || true
  _docker inspect "${CONTAINER}" > "${RESULT_ROOT}/container_inspect.json" 2>/dev/null || true
  _docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  _docker network rm "${EVAL_NET}" >/dev/null 2>&1 || true
}
trap cleanup EXIT
_docker network create "${EVAL_NET}" >/dev/null

if [[ "${GPU}" == "all" ]]; then
  GPU_SPEC="all"
elif [[ "${GPU}" =~ ^[0-9]+$ ]]; then
  GPU_SPEC="device=${GPU}"
else
  echo "GPU는 all 또는 GPU 번호 하나여야 합니다: ${GPU}" >&2
  exit 2
fi

# 규정 예시와 같이 image 뒤에 command/argument를 붙이지 않는다.
# --network 는 규정 §2의 오케스트레이터 호출과 같고, -p 는 §10의 로컬 검증 예시와 같다.
_docker run -d --name "${CONTAINER}" --network "${EVAL_NET}" --gpus "${GPU_SPEC}" \
  -p "127.0.0.1:${PORT}:8000" "${IMAGE_ID}" \
  > "${RESULT_ROOT}/container_id.txt"

BASE_URL="http://127.0.0.1:${PORT}"
deadline=$((SECONDS + STARTUP_TIMEOUT))
health_code=""
while (( SECONDS < deadline )); do
  if [[ "$(_docker inspect "${CONTAINER}" --format '{{.State.Running}}')" != "true" ]]; then
    _docker logs "${CONTAINER}" >&2 || true
    echo "모델/CUDA load 중 container가 종료됐습니다" >&2
    exit 1
  fi
  health_code="$(curl -sS --max-time 5 -o "${RESULT_ROOT}/health.json" \
    -w '%{http_code}' "${BASE_URL}/health" 2>/dev/null || true)"
  [[ "${health_code}" == "200" ]] && break
  sleep 5
done
if [[ "${health_code}" != "200" ]]; then
  _docker logs "${CONTAINER}" >&2 || true
  echo "/health가 ${STARTUP_TIMEOUT}초 안에 200이 되지 않았습니다" >&2
  exit 1
fi

# 규정 §1~2, §11-1. 평가 서버는 host port가 아니라 **같은 docker network 안에서
# 컨테이너 이름**으로 세 endpoint를 부른다. 서버가 127.0.0.1에만 bind되어 있으면
# 여기서만 실패하므로, 실제 평가 토폴로지를 그대로 재현해 확인한다.
echo "== eval-net 안에서 컨테이너 이름으로 접근 확인 (규정 §2) =="
# 외부 이미지를 새로 받지 않도록 제출 이미지 자체를 client로 재사용한다. 여기서는
# 서버가 아니라 오케스트레이터 역할이므로 entrypoint를 덮어써도 계약과 무관하다.
# `-i` 없이는 heredoc이 컨테이너 stdin에 붙지 않아 python이 즉시 EOF로 끝나고
# exit 0을 낸다. 즉 아무것도 검사하지 않은 채 통과한다. 반드시 필요하다.
_docker run --rm -i --network "${EVAL_NET}" --entrypoint python "${IMAGE_ID}" - \
  "http://${CONTAINER}:8000" <<'PY' > "${RESULT_ROOT}/in_network_probe.json" || {
import json, sys, urllib.request
base = sys.argv[1]
out = {"base_url": base}
for name, path in (("health", "/health"), ("models", "/v1/models")):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        out[name] = {"status": response.status, "body": json.loads(response.read().decode("utf-8"))}
    if out[name]["status"] != 200:
        raise SystemExit(f"{path} status={out[name]['status']}")
data = out["models"]["body"].get("data") or []
if not data or not str(data[0].get("id", "")).strip():
    raise SystemExit("/v1/models data[0].id가 비었습니다")
out["model_id"] = data[0]["id"]
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    echo "컨테이너 이름 http://${CONTAINER}:8000 접근 실패." >&2
    echo "서버가 0.0.0.0:8000이 아니라 127.0.0.1:8000에만 bind되었을 가능성이 큽니다(규정 §11-1)." >&2
    exit 1
  }
echo "   /health, /v1/models 모두 컨테이너 이름으로 접근 가능"

MODEL_ID="$(python3 - "${BASE_URL}" "${RESULT_ROOT}/models.json" <<'PY'
import json
import pathlib
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1] + "/v1/models", timeout=30) as response:
    payload = json.load(response)
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
model_id = payload["data"][0]["id"]
if not isinstance(model_id, str) or not model_id:
    raise SystemExit("/v1/models data[0].id가 비었습니다")
print(model_id)
PY
)"

# 규정의 marker 없는 일반 요청도 OpenAI Chat Completions envelope여야 한다.
python3 - "${BASE_URL}" "${MODEL_ID}" "${RESULT_ROOT}/generic_smoke.json" <<'PY'
import json
import pathlib
import sys
import urllib.request

payload = {
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": "안녕하세요. 한 줄로 자기소개해 주세요."}],
    "max_tokens": 64,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
}
request = urllib.request.Request(
    sys.argv[1] + "/v1/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=60) as response:
    result = json.load(response)
content = result["choices"][0]["message"]["content"]
if not isinstance(content, str) or not content:
    raise SystemExit("choices[0].message.content가 비었습니다")
pathlib.Path(sys.argv[3]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

# 짧은 예시만 통과하는 서버를 걸러내기 위해, 실제 공지와 같은 단일 user/marker
# 형식으로 12,000자 에세이 한 건을 보낸다. 점수 JSON과 세 근거까지 공식 parser로 확인한다.
python3 - "${BASE_URL}" "${MODEL_ID}" "${RESULT_ROOT}/long_input_smoke.json" \
  "${HERE}" "${REQUEST_TIMEOUT}" <<'PY'
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, sys.argv[4])
from official_parser_20260715 import _parse_model_output

essay = ("주장을 뒷받침하는 이유와 구체적인 사례를 제시한다. " * 1000)[:12000]
user_content = (
    "다음 글의 내용, 조직, 표현을 각각 1~5 정수로 평가하고 근거를 제시하라."
    "\n\n[prompt_text]\n학교에서 인공지능 활용을 허용해야 하는가?"
    "\n\n[essay_text]\n" + essay
)
payload = {
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": user_content}],
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "stop": ["Q:", "User:"],
}
request = urllib.request.Request(
    sys.argv[1] + "/v1/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=int(sys.argv[5])) as response:
    result = json.load(response)
content = result["choices"][0]["message"]["content"]
parsed = _parse_model_output(content)
if parsed is None:
    raise SystemExit("12,000자 입력 응답이 공식 JSON parser를 통과하지 못했습니다")
for trait in ("content", "organization", "expression"):
    score = parsed[trait]["score"]
    rationale = parsed[trait].get("rationale")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise SystemExit(f"12,000자 입력의 {trait} 점수가 1~5 정수가 아닙니다: {score!r}")
    if not isinstance(rationale, str) or not rationale.strip():
        raise SystemExit(f"12,000자 입력의 {trait} 근거가 비었습니다")
pathlib.Path(sys.argv[3]).write_text(
    json.dumps(
        {"essay_chars": len(essay), "request": payload, "response": result},
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY

EVAL_ARGS=(
  --base-url "${BASE_URL}"
  --input "${DATA_FILE}"
  --output-dir "${RESULT_ROOT}/evaluation"
  --expected-count 400
  --readiness-timeout 30
  --request-timeout "${REQUEST_TIMEOUT}"
  --request-max-tokens 512
  --expected-model "${MODEL_ID}"
  --image-ref "${IMAGE_REF}"
  --container-name "${CONTAINER}"
  --gpu-spec "${GPU}"
)
[[ -z "${LIMIT}" ]] || EVAL_ARGS+=(--limit "${LIMIT}")
python3 "${HERE}/evaluate_http.py" "${EVAL_ARGS[@]}" \
  2>&1 | tee "${RESULT_ROOT}/evaluation.log"

# full 400편에서는 release가 고정한 HTTP oracle과 C/O/E 정수가 한 건도 달라지지 않아야 한다.
if [[ -z "${LIMIT}" ]]; then
  python3 - "${RESULT_ROOT}/evaluation/records.jsonl" \
    "${RESULT_ROOT}/evaluation/prediction_hash.json" \
    "${EXPECTED_PREDICTION_SHA256}" <<'PY'
import hashlib
import json
import pathlib
import sys

expected = sys.argv[3]
records_path = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
canonical = "".join(
    f"{row['essay_id']}\t{row['official_scores']['content']}\t"
    f"{row['official_scores']['organization']}\t"
    f"{row['official_scores']['expression']}\n"
    for row in rows
).encode("utf-8")
actual = hashlib.sha256(canonical).hexdigest()
report = {
    "format": "essay_id\\tcontent_int\\torganization_int\\texpression_int\\n",
    "count": len(rows),
    "expected_sha256": expected,
    "actual_sha256": actual,
    "match": actual == expected,
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
if len(rows) != 400 or actual != expected:
    raise SystemExit(f"prediction parity 실패: {report}")
print(f"prediction parity PASS: {actual}")
PY
fi

# 조용한 오채점 게이트. 서버가 요청에서 뽑아낸 prompt/essay가 의심스러우면
# `serve.py`가 container.log에 고정 토큰을 남긴다. 응답만 보는 다른 게이트는
# 이 실패를 절대 잡지 못한다(200 + 정상 파싱 + degradation 없음).
_docker logs "${CONTAINER}" > "${RESULT_ROOT}/container.log" 2>&1 || true
MARKER_HITS="$(grep -c 'REQUEST_MARKER_ANOMALY' "${RESULT_ROOT}/container.log" || true)"
if [[ "${MARKER_HITS}" != "0" ]]; then
  grep 'REQUEST_MARKER_ANOMALY' "${RESULT_ROOT}/container.log" | head -5 >&2
  echo "요청 마커 이상이 ${MARKER_HITS}건 있습니다. 다른 글을 채점했을 수 있습니다." >&2
  exit 1
fi
# 강등이 하나라도 있으면 로컬에서 고칠 수 있는 버그다. 평가 서버에서는 강등이
# 0점보다 낫지만, 로컬에서 나왔다면 그대로 내보내면 안 된다.
DEGRADED="$(grep -c '강등된 응답' "${RESULT_ROOT}/container.log" || true)"
# 2026-08-25. greedy 반복 루프로 근거 JSON이 안 닫히는 편이 400편 중 3편 있다.
# engine에 표본추출 복구 경로가 있지만 76~81초가 필요하고(직접 재현 측정), deadline을
# 그만큼 늘리면 주최측 미지의 timeout에 걸려 그 편이 **0점**이 될 위험이 생긴다.
# 한 행을 0으로 잃는 비용이 강등 비용의 30배이므로 deadline을 낮게 묶고 강등을 받아들인다.
# 강등돼도 **점수는 응답에 온전히 실린다**(engine.respond). 컨테이너 400편 전수 대조에서
# 강등 3편의 점수가 로컬 앙상블과 정확히 일치함을 확인했다.
# 근본 해법으로 repetition_penalty를 400편 전수 시험했으나 루프를 옮기기만 하고
# (1.05: 실패 3->2, 겹침 0) 근거성이 192->169/136편으로 나빠져 기각했다.
# 기본값 0이라 이 변수를 명시하지 않으면 기존과 완전히 같은 판정을 한다.
DEGRADED_BUDGET="${HTTP_DEGRADED_RESPONSE_BUDGET:-0}"
if [[ "${DEGRADED}" -gt "${DEGRADED_BUDGET}" ]]; then
  grep '강등된 응답' "${RESULT_ROOT}/container.log" | head -5 >&2
  echo "강등된 응답이 ${DEGRADED}건 있습니다(허용치 ${DEGRADED_BUDGET}). 배포 전에 원인을 제거하세요." >&2
  exit 1
fi
echo "요청 마커 이상 0건 / 강등 응답 ${DEGRADED}건 (허용치 ${DEGRADED_BUDGET})"

echo "PASS: ${IMAGE_REF} (${IMAGE_ID})"
echo "결과: ${RESULT_ROOT}"
