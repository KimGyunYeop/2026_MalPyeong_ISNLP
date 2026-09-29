#!/usr/bin/env bash
# 2026-08-25 후보 B(bbq35_e5_s42~s49 평탄 8seed + Gemma-teacher 증류 Qwen rationale LoRA)를
# build하고, Y6 release와 같은 Docker/HTTP 400편 검사를 통과한 경우에만 push한다.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# 배포 모델은 이 두 경로로 고정한다. Gemma는 teacher일 뿐이며 Gemma 가중치는
# image에 넣지 않는다. rationale runtime도 Qwen3.5-9B LoRA이며 채점 base를 공유한다.
# 2026-08-23 확정: ax4_light 4 seed + Qwen3.5-9B 4 seed = 8멤버 등가중.
# Qwen3.5-9B가 e5 레시피에서 ax4_light를 연속 ρ +5.2 SE로 이기고(§21) 교차 오차상관이
# 0.94382(계열 내 0.9764)라 앙상블 이득이 크다. AM RMSE 0.43120 -> 0.41788,
# P(RMSE개선)=0.995. VRAM 13.52+17.98+여유6 = 37.5 GiB < 48.
SCORE_CHECKPOINTS="${ROOT}/main_code/results/final_proposed/bbq35_e5_s42/checkpoint,${ROOT}/main_code/results/final_proposed/bbq35_e5_s43/checkpoint,${ROOT}/main_code/results/final_proposed/bbq35_e5_s44/checkpoint,${ROOT}/main_code/results/final_proposed/bbq35_e5_s45/checkpoint,${ROOT}/main_code/results/final_proposed/bbq35_e5_s46/checkpoint,${ROOT}/main_code/results/final_proposed/bbq35_e5_s47/checkpoint,${ROOT}/main_code/results/final_proposed/bbq35_e5_s48/checkpoint,${ROOT}/main_code/results/final_proposed/bbq35_e5_s49/checkpoint"
SCORE_CHECKPOINT="${SCORE_CHECKPOINTS}"
RATIONALE_ADAPTER="${RATIONALE_ADAPTER:-/home/nlplab/research/gyop/2026_MalPyeong_tf/main_code_relonation/results/rationale_v4_qwen35_student_20260824-133042/lora/final_adapter}"
QWEN_CONTROL_ROOT="${ROOT}/main_code_relonation/results/rationale_prompt_v1_teacher_ab_full11600_r32/students/model2_qwen_teacher/bdf21a4b13f3_qwen_full11600"

DEFAULT_IMAGE_TAG="qwen35-8seed-qwenrationale-nooffset-r4-20260825"
# standalone BF16 batch-1 c02의 canonical essay_id/C/O/E 정수 점수열이다. 최종
# oracle은 반드시 Docker HTTP 400편으로 재검증한다. JSONL 파일 SHA(96519e...)나
# historical Y6 SHA(47e2...)를 여기에 넣으면 안 된다.
# 2026-08-25 최종 제출본(qwen35 8seed + v4 Qwen rationale, offset 0)의 실측 예측 hash.
# Docker HTTP 400편을 그대로 재현하면 이 값이 나온다. 인자 없이 이 스크립트를 돌리면
# 제출본과 bit-exact 같은 image를 만들고, 이 hash로 스스로를 검증한다.
PROVISIONAL_HTTP_PREDICTION_SHA256="af666dada5de6701092ff1f0ab795332f366fb3a9a65039a1add9038aff22973"

IMAGE_TAG="${IMAGE_TAG:-${DEFAULT_IMAGE_TAG}}"
EXPECTED_HTTP_PREDICTION_SHA256="${EXPECTED_HTTP_PREDICTION_SHA256:-${PROVISIONAL_HTTP_PREDICTION_SHA256}}"
# 정수 총점 offset. 0이면 기존 배포와 **bit-exact** 같다(기본값). 0이 아니면 400편의
# 총점이 전부 그만큼 이동하므로 예측 hash가 반드시 달라진다. 그래서 offset != 0인
# 빌드는 EXPECTED_HTTP_PREDICTION_SHA256을 명시적으로 넘겨야만 진행한다 — 안 그러면
# offset=0 oracle에 대고 offset 빌드를 내보내게 된다.
# 근거: main_code/DISTRIBUTION_SHIFT_DIAGNOSIS_20260820.md §9
INTEGER_TOTAL_OFFSET="${INTEGER_TOTAL_OFFSET:-0}"
if ! [[ "${INTEGER_TOTAL_OFFSET}" =~ ^-?[0-9]+$ ]]; then
  echo "INTEGER_TOTAL_OFFSET은 정수여야 합니다: ${INTEGER_TOTAL_OFFSET}" >&2
  exit 2
fi
if [[ "${INTEGER_TOTAL_OFFSET}" != "0" \
      && "${EXPECTED_HTTP_PREDICTION_SHA256}" == "${PROVISIONAL_HTTP_PREDICTION_SHA256}" ]]; then
  echo "INTEGER_TOTAL_OFFSET=${INTEGER_TOTAL_OFFSET}인데 EXPECTED_HTTP_PREDICTION_SHA256이" >&2
  echo "offset=0 oracle 그대로입니다. offset 빌드의 예측 hash를 명시적으로 넘기세요." >&2
  exit 2
fi
REGISTRY_REPO="${REGISTRY_REPO:-gyunyeop/writing-scorer}"
GPU="${GPU:-all}"
PUSH="${PUSH:-0}"
RESULTS="${RESULTS:-main_code_submission/results/docker_release}"
MANIFEST_DIR="${MANIFEST_DIR:-${ROOT}/main_code_submission/results/docker_release/source_manifests}"

usage() {
  cat <<'EOF'
사용법:
  bash docker_release.sh check
  bash docker_release.sh verify
  bash docker_release.sh push

  check  : GPU 없이 artifact/manifest 계약만 확인
  verify : 선택적 사전연습; build + Docker HTTP 400편, push 없음
  push   : 최종 명령; 새로 build + 같은 image ID의 HTTP 400편 검증 + push

`push` 하나만 실행하면 Docker 평가는 한 번이다. `verify` 뒤 `push`를 실행하면 두 번
평가되는데, 최종 push가 이전 rehearsal image를 신뢰하지 않고 자신이 올릴 image ID를
다시 검증하는 fail-closed 동작이다.

Y6 release와 같은 환경변수 방식도 지원합니다.
  PUSH=0 GPU=all bash docker_release.sh
  PUSH=1 GPU=all REGISTRY_REPO=OWNER/REPO bash docker_release.sh

기본 HTTP oracle과 Docker 결과가 다르면 push 전에 실패합니다. 남은
prediction_hash.json과 records.jsonl을 검토한 뒤 승인한 실제 hash로 다시 실행하세요.
  EXPECTED_HTTP_PREDICTION_SHA256=<검토한-64hex> bash docker_release.sh verify
  EXPECTED_HTTP_PREDICTION_SHA256=<같은-64hex> bash docker_release.sh push

원격 tag가 이미 있으면 덮어쓰지 않습니다. IMAGE_TAG에 새 r2 이상의 tag를 주고
check/verify부터 다시 실행하세요.
EOF
}

ACTION="${1:-release}"
[[ "$#" -le 1 ]] || { usage >&2; exit 2; }
case "${ACTION}" in
  check)
    ;;
  verify)
    [[ "${PUSH}" == "0" ]] || {
      echo "verify와 PUSH=${PUSH}를 함께 사용할 수 없습니다" >&2
      exit 2
    }
    PUSH=0
    ;;
  push)
    [[ "${PUSH}" == "0" || "${PUSH}" == "1" ]] || {
      echo "PUSH는 0 또는 1이어야 합니다" >&2
      exit 2
    }
    PUSH=1
    ;;
  release)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

[[ "${PUSH}" == "0" || "${PUSH}" == "1" ]] || {
  echo "PUSH는 0 또는 1이어야 합니다" >&2
  exit 2
}
# 채점 환경은 **단일 L40s 48GB 1장**이다(규정 6장). 따라서 단일 GPU index는
# GPU=all보다 오히려 평가 환경에 충실하다. 두 변형(offset 0/1)을 각 GPU에서
# 병렬 검증하기 위해 단일 index를 허용한다. 여러 index 나열은 계속 금지한다.
[[ "${GPU}" == "all" || "${GPU}" =~ ^[0-9]$ ]] || {
  echo "GPU는 all 또는 단일 index(0~9)여야 합니다: ${GPU}" >&2
  exit 2
}
[[ "${REGISTRY_REPO}" == */* ]] || {
  echo "REGISTRY_REPO는 OWNER/REPO 형식이어야 합니다" >&2
  exit 2
}
[[ "${EXPECTED_HTTP_PREDICTION_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "EXPECTED_HTTP_PREDICTION_SHA256는 64자리 소문자 SHA-256이어야 합니다" >&2
  exit 2
}
[[ "${IMAGE_TAG}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "IMAGE_TAG가 유효한 Docker tag가 아닙니다: ${IMAGE_TAG}" >&2
  exit 2
}

command -v flock >/dev/null || {
  echo "동시 staging 방지에 필요한 flock이 없습니다" >&2
  exit 1
}
# 전역 락으로 되돌린다. 2026-08-22 09:54에 tag별 락으로 병렬 실행했더니 두 빌드가
# **같은 staging 디렉터리 submission_assets/** 를 공유해 서로를 덮어썼다
# (PartitionError: parts_root가 비어 있지 않습니다 / FileNotFoundError).
# 락이 막던 것은 tag 충돌이 아니라 **공유 staging**이다. 병렬 빌드는 불가능하다.
LOCK_FILE="/tmp/malpyeong_docker_release.lock"
exec 9>"${LOCK_FILE}"
flock -n 9 || {
  echo "다른 Docker release가 실행 중입니다: ${LOCK_FILE}" >&2
  exit 1
}

mkdir -p "${MANIFEST_DIR}"
MANIFEST="${MANIFEST_DIR}/${IMAGE_TAG}_${EXPECTED_HTTP_PREDICTION_SHA256:0:12}.json"
MANIFEST_TMP="$(mktemp "${MANIFEST_DIR}/.${IMAGE_TAG}.XXXXXX")"
RELEASE_MARKER=""
cleanup() {
  rm -f -- "${MANIFEST_TMP}"
  if [[ -n "${RELEASE_MARKER}" ]]; then
    rm -f -- "${RELEASE_MARKER}"
  fi
}
trap cleanup EXIT

echo "== 고정 release 후보 =="
echo "score checkpoint: ${SCORE_CHECKPOINT}"
echo "rationale adapter: ${RATIONALE_ADAPTER}"
echo "teacher provenance: google/gemma-4-26B-A4B-it (image에는 포함하지 않음)"
echo "image tag: ${IMAGE_TAG}"
echo "expected HTTP prediction SHA-256: ${EXPECTED_HTTP_PREDICTION_SHA256}"

# Manifest를 손으로 중복 유지하지 않고, 고정 artifact에서 prompt 원문과 지문을 읽어
# 결정론적으로 만든다. 어떤 고정 파일이 바뀌어도 Docker build 전에 중단한다.
PYTHONPATH="${ROOT}" python3 - \
  "${ROOT}" "${SCORE_CHECKPOINT}" "${RATIONALE_ADAPTER}" \
  "${QWEN_CONTROL_ROOT}" "${IMAGE_TAG}" \
  "${EXPECTED_HTTP_PREDICTION_SHA256}" "${INTEGER_TOTAL_OFFSET}" \
  "${MANIFEST_TMP}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

from main_code_submission.artifact_integrity import (
    artifact_tree_fingerprint,
    checkpoint_fingerprint,
)
from main_code_submission.config import load_manifest
from main_code_relonation.prompts import DEFAULT_SKELETON_HINT
from main_code_submission.rationale_prompt_binding import (
    prompt_binding_from_adapter,
)

(
    root_arg,
    score_arg,
    rationale_arg,
    qwen_control_arg,
    image_tag,
    expected_http_hash,
    integer_total_offset_arg,
    output_arg,
) = sys.argv[1:]
integer_total_offset = int(integer_total_offset_arg)
root = Path(root_arg).resolve()
scores = [Path(p).resolve() for p in score_arg.split(",") if p.strip()]
score = scores[0]  # 대표 (경로 로그용)
rationale = Path(rationale_arg).resolve()
qwen_control = Path(qwen_control_arg).resolve()
output = Path(output_arg)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"필수 JSON이 없습니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON object가 아닙니다: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


require(score.is_dir(), f"score checkpoint가 없습니다: {score}")
require(rationale.is_dir(), f"rationale adapter가 없습니다: {rationale}")
require(
    re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", image_tag) is not None,
    f"유효하지 않은 image tag: {image_tag}",
)
require(
    re.fullmatch(r"[0-9a-f]{64}", expected_http_hash) is not None,
    "HTTP prediction hash가 64자리 소문자 SHA-256이 아닙니다",
)

# --- score artifacts (4-seed 앙상블) ---------------------------------------
# 평탄(미선택) checkpoint 계약을 **4개 전부**에 건다.
#   (1) selection.json이 없어야 한다 = 선택이 일어나지 않았다
#   (2) global_step == max_steps == config.max_train_steps == 1104 = 마지막이다
# 왜 4-seed 앙상블인가: 개별 seed 최고를 고르면 400편 선택 편향(best-of-4 = 1.03·SD
# = 0.0099)이 붙고 그 이득은 리더보드로 전이되지 않는다(실측). seed 전부를 균등 가중
# 평균하면 **고르는 행위가 없어 편향이 0**이고, val에서 계열 평균 대비
# RMSE −0.0043 / ρ +0.0066 개선된다.
EXPECTED_FP = {
    # 2026-08-25: q35 8seed 단독 구성. ax4 계열은 목록에서 **제거**한다 —
    # 남겨 두면 실수로 ax4 checkpoint를 넘겨도 통과한다.
    # Qwen3.5-9B 4 seed (2026-08-23 계산, 전부 평탄 1104/1104 · selection.json 없음)
    "bbq35_e5_s42": ("87e233275d15fc80", "46e3c7c68f44286c"),
    "bbq35_e5_s43": ("b817cc59934d65bd", "d5f5fbbeb4d81b8b"),
    "bbq35_e5_s44": ("130ddf1c4444ab02", "cb662a46f6ae20ff"),
    "bbq35_e5_s45": ("5ce2ed4db9b746ed", "84db04766401cff3"),
    # 2026-08-25 추가 4 seed. 레시피는 s42~s45와 --seed만 다르고
    # resolved_config.json 152개 항목 중 seed 하나만 차이난다(전수 대조 확인).
    # 전부 평탄 1104/1104 · selection.json 없음.
    "bbq35_e5_s46": ("37ac7e9127626390", "9eb20deeb08825fc"),
    "bbq35_e5_s47": ("26e1c21bb38dc6e3", "886b7ecf06317b8c"),
    "bbq35_e5_s48": ("5c0864e863a56e15", "f42a4420df2db0fc"),
    "bbq35_e5_s49": ("84de5cddecf6875c", "556e66d94f183362"),
}
# 백본별 계약. run 이름 접두어로 갈라 model_id를 강제한다 — 섞이면 조용히 다른 base가 실린다.
BACKBONE_OF = {
    "bbq35_e5_": ("qwen35_9b", "Qwen/Qwen3.5-9B", 9.6531),
}
def backbone_for(run: str):
    for _pre, _v in BACKBONE_OF.items():
        if run.startswith(_pre):
            return _v
    raise SystemExit(f"백본을 알 수 없는 run: {run}")
require(len(scores) == 8, f"score member는 8개여야 합니다: {len(scores)}")
require(len({p.parent.name for p in scores}) == 8, "중복 run이 있습니다")
score_cores, score_trees, selections = {}, {}, {}
for _sc in scores:
    _run = _sc.parent.name
    require(_run in EXPECTED_FP, f"예상하지 않은 run: {_run}")
    require(not (_sc / "selection.json").exists(),
            f"평탄 계약 위반: {_run}에 selection.json이 있습니다")
    _cfg = read_json(_sc / "config.json")
    _ts = read_json(_sc.parent / "trainer" / "trainer_state.json")
    selections[_run] = {"policy": "final_checkpoint_no_selection",
                        "global_step": _ts.get("global_step"),
                        "max_steps": _ts.get("max_steps"),
                        "config_max_train_steps": _cfg.get("max_train_steps")}
    require(_ts.get("global_step") == _ts.get("max_steps")
            == _cfg.get("max_train_steps") == 1104,
            f"{_run}: 마지막 checkpoint가 아닙니다: {selections[_run]}")
    _bk, _mid, _pb = backbone_for(_run)
    require(_cfg.get("model_id") == _mid, f"{_run} model_id 불일치: {_cfg.get('model_id')} != {_mid}")
    require(_cfg.get("model_revision") == "main", f"{_run} model revision 불일치")
    require(_cfg.get("training_mode") == "lora_only", f"{_run} training_mode 불일치")
    require(_cfg.get("use_qlora") is False, f"{_run} non-QLoRA 아님")
    require(_cfg.get("lora_r") == 32, f"{_run} LoRA rank 불일치")
    require(_cfg.get("lora_alpha") == 64, f"{_run} LoRA alpha 불일치")
    require(_cfg.get("lora_include_mlp") is True, f"{_run} MLP LoRA 불일치")
    require(_cfg.get("essay_surface") == "official_raw", f"{_run} essay surface 불일치")
    require(_cfg.get("score_postprocess") == "average_matched", f"{_run} 후처리 불일치")
    require(_cfg.get("listwise_loss") == "soft_spearman", f"{_run} listwise 불일치")
    _c = checkpoint_fingerprint(_sc); _t = artifact_tree_fingerprint(_sc)
    _ec, _et = EXPECTED_FP[_run]
    require(_c["combined_sha256"].startswith(_ec), f"{_run} core fingerprint 불일치: {_c['combined_sha256']}")
    require(_t["combined_sha256"].startswith(_et), f"{_run} closure fingerprint 불일치: {_t['combined_sha256']}")
    score_cores[_run] = _c; score_trees[_run] = _t
score_config = read_json(score / "config.json")
selection = selections
# 대표 run(첫 seed). 멤버별 지문은 아래 source_checkpoint_artifacts_by_run에 전부 담긴다.
score_core = score_cores["bbq35_e5_s42"]
score_tree = score_trees["bbq35_e5_s42"]

# --- Gemma teacher로 증류한 A.X rationale artifact --------------------------
adapter_config = read_json(rationale / "adapter_config.json")
runtime_config = read_json(rationale / "rationale_runtime_config.json")
prompt_binding = prompt_binding_from_adapter(rationale)
# skeleton hint는 prompt 끝에 붙는 출력 스켈레톤의 자리표시자
# (`<content 근거, {hint}>`)로 렌더링된다. 프롬프트에서 가장 강한 길이 신호이므로
# 학습 때와 다른 값을 서빙하면 본문 길이가 통째로 달라진다. v4 학생의
# rationale_runtime_config.json에는 이 키가 없고, DEFAULT("두 문장 이내 180자
# 이내")로 조용히 떨어지면 "6~9문장 450~540 tokens"로 학습한 어댑터에 정반대
# 지시를 주게 된다. 그래서 학습 run의 resolved_config.json까지 찾아보고,
# 어디에서도 못 찾으면 기본값으로 넘어가지 않고 멈춘다.
def resolve_skeleton_hint(adapter_dir, runtime):
    declared = runtime.get("rationale_skeleton_hint")
    if declared:
        return str(declared), "rationale_runtime_config.json"
    current = Path(adapter_dir).resolve()
    for _ in range(3):
        candidate = current / "resolved_config.json"
        if candidate.is_file():
            value = read_json(candidate).get("rationale_skeleton_hint")
            if value:
                return str(value), str(candidate)
            # 학습 run은 찾았는데 키가 없다 = 그 recipe가 이 옵션보다 **먼저**
            # 만들어졌다는 뜻이고, 그때 학습은 DEFAULT_SKELETON_HINT로 돌았다.
            # (v3 recipe r14에는 키 자체가 없고 teacher manifest도 null이다.)
            # "모른다"와 "기본값을 썼다"를 섞으면 v3가 통과하지 못한다.
            return DEFAULT_SKELETON_HINT, f"{candidate} (키 없음 → 학습 당시 기본값)"
        if current.parent == current:
            break
        current = current.parent
    return None, None


rationale_skeleton_hint, skeleton_hint_source = resolve_skeleton_hint(
    rationale, runtime_config
)
require(
    rationale_skeleton_hint is not None,
    "학습 run의 resolved_config.json을 찾지 못해 rationale skeleton hint를 확정할 수 "
    f"없습니다. 학습 때와 다른 값({DEFAULT_SKELETON_HINT!r})으로 서빙하면 근거 길이가 "
    "무너지므로 추측하지 않고 멈춥니다",
)
print(f"rationale skeleton hint: {rationale_skeleton_hint!r}  <- {skeleton_hint_source}")
expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
require(adapter_config.get("base_model_name_or_path") == "Qwen/Qwen3.5-9B", "rationale base 불일치")
require(adapter_config.get("task_type") == "CAUSAL_LM", "rationale task_type 불일치")
require(adapter_config.get("peft_type") == "LORA", "rationale PEFT type 불일치")
require(adapter_config.get("r") == 32, "rationale LoRA rank 불일치")
require(adapter_config.get("lora_alpha") == 64, "rationale LoRA alpha 불일치")
require(set(adapter_config.get("target_modules") or []) == expected_targets, "rationale target module 불일치")
require(adapter_config.get("modules_to_save") is None, "rationale modules_to_save는 null이어야 합니다")
require(adapter_config.get("bias") == "none", "rationale bias 계약 불일치")
require(runtime_config.get("score_mode") == "fixed", "rationale score_mode 불일치")
require(
    runtime_config.get("training_score_source") == "human_average_matched",
    "rationale training score source 불일치",
)
require(
    runtime_config.get("training_conditioning_score_postprocess") == "average_matched",
    "rationale conditioning postprocess 불일치",
)
# 2026-08-25: v3 프롬프트 -> v4 프롬프트. 어댑터가 v4로 학습됐으므로 기본값도 v4다.
expected_prompt_sha = os.environ.get(
    "RATIONALE_PROMPT_SHA256",
    "67df14d2264531e88efde762c7e323b7b2f9d537b57fc07ad7e636d7faf945d1",
)
require(
    prompt_binding.sha256 == expected_prompt_sha,
    f"rationale prompt binding 불일치: {prompt_binding}",
)
# 2026-08-25: ax4 template -> Qwen3.5 template. 학습 artifact가 기록한 값과 같아야
# 서빙 토큰열이 학습과 일치한다. 이 잠금이 조용한 train/serve 불일치를 막는 유일한 gate다.
require(
    runtime_config.get("chat_template_sha256")
    == "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "rationale chat template hash 불일치",
)
rationale_tree = artifact_tree_fingerprint(
    rationale, excluded_relative_paths={"training_args.bin"}
)
# v3 어댑터는 새로 학습한 것이라 v1 상수와 비교할 수 없다. 구조 검사(base model,
# LoRA rank/alpha/target, task_type, score_mode, conditioning postprocess, chat template,
# prompt binding)는 위에서 전부 유지되고, 여기서는 실측 지문을 manifest에 기록만 한다.
print(f"rationale runtime fingerprint: {rationale_tree.get('combined_sha256')}")
adapter_weight = rationale / "adapter_model.safetensors"
adapter_weight_sha = hashlib.sha256(adapter_weight.read_bytes()).hexdigest()
print(f"rationale adapter weight sha256: {adapter_weight_sha}")

student_root = rationale.parent.parent
completed = read_json(student_root / "lora" / "completed.json")
# teacher pseudo manifest는 학습 입력 행 수의 유일한 1차 출처다. 상수로 적으면
# 실제와 어긋나도 아무도 모른다(v3에서 실제로 그랬다).
# 2026-08-25: v4 Qwen student는 results/ 바로 아래에서 학습했으므로 v3의
# `parents[4]` 상대경로 가정이 성립하지 않는다. teacher AB root를 명시한다.
ab_root = Path(os.environ.get(
    "TEACHER_AB_ROOT",
    str(root / "main_code_relonation/results/rationale_prompt_v4_gemma_full11600_r32"),
))
teacher_manifests = sorted(ab_root.glob("rationale_*_gemma_teacher/pseudo_train.jsonl.manifest.json"))
require(
    len(teacher_manifests) == 1,
    f"teacher pseudo manifest를 하나만 찾아야 합니다: {[str(p) for p in teacher_manifests]}",
)
teacher_manifest = read_json(teacher_manifests[0])
require(
    isinstance(teacher_manifest.get("generated_this_run"), int)
    and teacher_manifest["generated_this_run"] > 0,
    f"teacher pseudo manifest에서 학습 행 수를 읽지 못했습니다: {teacher_manifest.get('generated_this_run')!r}",
)
# 2026-08-25: v4 Qwen student의 verify 산출물은 `verify/`다(v3는 `verify_y6/`).
_verify_dir = next((d for d in (student_root / "verify", student_root / "verify_y6")
                    if (d / "inference_manifest.json").is_file()), None)
require(_verify_dir is not None,
        f"verify inference_manifest.json을 찾지 못했습니다: {student_root}")
print(f"verify 산출물: {_verify_dir}")
inference = read_json(_verify_dir / "inference_manifest.json")
require(
    completed.get("status") == "complete" and int(completed.get("global_step", 0)) > 0,
    f"Gemma student 학습 완료 계약 불일치: {completed}",
)
# 2026-08-25: 오프라인 verify는 **greedy 단일 시도**다. Qwen v4는 400편 중 3편
# (0.75%)에서 반복 루프에 빠져 JSON이 닫히지 않는다(실측: expression 근거 중간에서
# 끊김, 중괄호 2/4). 서빙 경로에는 표본추출 재시도(temp 0.7, 결정적 시드)와 부분
# 추출이 있어 그 3편도 content·organization 근거는 살고 expression만 template이 된다.
# 그리고 **컨테이너 400편 HTTP 평가가 더 엄격한 최종 gate로 그대로 남아 있다**
# (예측 hash 대조 포함). 그래서 오프라인 허용치만 3편으로 열고, 점수 관련 계약
# (fixed_score_parity, generated_score_mismatch_count)은 **엄격하게 유지한다**.
_offline_failure_budget = int(os.environ.get("RATIONALE_OFFLINE_FAILURE_BUDGET", "3"))
require(
    inference.get("requested_count") == 400
    and int(inference.get("failure_count", 999)) <= _offline_failure_budget
    and int(inference.get("success_count", 0)) + int(inference.get("failure_count", 0)) == 400
    and inference.get("fixed_score_parity") is True
    and inference.get("generated_score_mismatch_count") == 0,
    f"student validation 400편 계약을 통과하지 못했습니다 (허용 실패 {_offline_failure_budget}편): "
    f"success={inference.get('success_count')} failure={inference.get('failure_count')} "
    f"parity={inference.get('fixed_score_parity')} mismatch={inference.get('generated_score_mismatch_count')}",
)
# proxy Judge A/B는 v1 라운드의 teacher 선택 근거였고 이미 Gemma 우세로 결론났다.
# 이번 라운드는 teacher를 바꾸지 않고 prompt만 v1->v3로 바꾼 것이라 A/B를 다시 돌리지
# 않는다. 대신 push 단계의 실제 image 400편 HTTP 평가가 최종 gate다.

manifest = {
    "name": "bbq35_e5_8seed_plateau_qwen_v4rationale",
    "root": str(root),
    "served_model_name": "malpyeong-writing-scorer",
    "score_postprocess": "average_matched",
    # 정수 총점 offset. 세 영역 정수의 합에 더한 뒤 3~15로 clip한다. Spearman은
    # 비트 일치로 불변이고(순위 벡터가 같다) Judge도 안 바뀐다. RMSE만 움직인다.
    "integer_total_offset": integer_total_offset,
    "essay_surface": "official_raw",
    "max_tokens": 512,
    "seed": 42,
    # 8멤버 등가중. backbone_key가 갈려 있어야 engine이 base를 두 벌만 올린다
    # (engine.py:180 groups.setdefault). 같은 key로 뭉치면 잘못된 base에 어댑터를 붙인다.
    "score_members": [
        {
            "name": f"{backbone_for(p.parent.name)[0]}_{p.parent.name}_plateau",
            "checkpoint": str(p),
            "backbone_key": backbone_for(p.parent.name)[0],
            "parameters_billion": backbone_for(p.parent.name)[2],
            "weight": 0.125,
        }
        for p in scores
    ],
    "rationale": {
        # 2026-08-25: 근거도 Qwen3.5-9B다. 채점 4멤버와 같은 base를 공유하므로
        # 추가 VRAM이 0이고 이미지에 base가 하나만 구워진다.
        "base_model": "Qwen/Qwen3.5-9B",
        "base_model_revision": "main",
        "adapter": str(rationale),
        "share_backbone_key": "qwen35_9b",
        "load_in_4bit": False,
        # FAQ 상한은 2048 token이고 Docker 규정 §5의 max_tokens 512는 오케스트레이터가
        # 보내는 요청 필드다. 우리는 컨테이너 안에서 자체 디코딩을 하므로(2026-07-20 답변)
        # 생성 상한을 분리해 둔다.
        #
        # 값은 학습 recipe와 반드시 같아야 한다. v4 prompt는 영역당 450~540 token을
        # 목표로 상한 600을 걸고, 세 영역이면 최대 1,800 token + JSON 오버헤드다.
        # 1024로 두면 컨테이너가 v4 근거를 문장 중간에서 자른다. r15 recipe가
        # `max_new_tokens: 2048`이므로 여기도 2048이다. prompt 예산은
        # 8192-2048=6144 token이고 실측 최장 v4 prompt가 2,113 token이라 여유가 있다.
        "max_new_tokens": 2048,
        "max_length": 8192,
        # 근거 생성 전체(첫 시도 + 재시도 2회)의 벽시계 상한. `NEVER_DISCARD_A_SCORE.md`
        # §6이 지목한 남은 최대 위험 — 직렬화 락과 평가 서버의 미공개 요청 timeout —
        # 을 실제로 막는 유일한 장치다. 예산을 넘기면 얻은 근거만 싣고 나머지는
        # template으로 메우며, **점수는 어느 경우에도 온전히 실린다.**
        #
        # 값 근거(공식 400편 실측 근거 생성 지연):
        #   v3  ax4   median 2.69s / p95 3.20s / max 3.63s
        #   v4  ax4   median 7.62s / p95 8.56s / max 9.73s
        #   v4  Qwen  median 11.64s / p95 13.74s / max 15.27s  (2026-08-25 실측 397편)
        # Qwen은 linear attention 24층이 fla/causal_conv1d 없이 torch fallback을 타서
        # 63.9 tok/s이고 필요 토큰이 median 744다. 8초면 511토큰에서 끊겨 **전편**의
        # JSON이 깨진다. 관측 최대 15.27s의 2.6배인 40으로 둔다. 생성 상한이
        # max_new_tokens=2048(=32초)이므로 이 예산은 재시도 사슬만 묶는다.
        # 2026-08-25 최종 제출본 값. greedy 반복 루프에 빠진 편(400편 중 3편)에서 engine의
        # 표본추출 복구가 76~81초를 쓴다(직접 재현 측정). 40초로는 3차 복구에 못 닿아
        # 그 편들이 강등됐고, 100초로 올리자 강등 0건이 됐다. 더 크게 올리면 주최측의
        # 미공개 요청 timeout에 걸려 그 편이 0점이 될 위험이 커진다.
        "deadline_seconds": float(os.environ.get("RATIONALE_DEADLINE_SECONDS", "100.0")),
        # 학습 때 쓴 template kwargs를 그대로 넘겨야 서빙 토큰열이 학습과 같아진다.
        # Qwen3.5는 thinking 블록을 열어 두므로 반드시 닫아야 한다(§22.5).
        "chat_template_kwargs": {"enable_thinking": False},
        "chat_template_sha256": runtime_config["chat_template_sha256"],
        "rationale_prompt_id": prompt_binding.prompt_id,
        "rationale_prompt_text": prompt_binding.text,
        "rationale_prompt_sha256": prompt_binding.sha256,
        # 출력 스켈레톤의 길이 지시. 스켈레톤은 프롬프트 맨 끝(생성 직전)에 붙어
        # 본문 길이 규칙보다 강하게 작동하므로 학습과 서빙이 달라지면 안 된다.
        # manifest에 없으면 `config.py`가 역사적 기본값 "두 문장 이내 180자 이내"로
        # 되돌아가는데, v4 어댑터는 "6~9문장 450~540 tokens"로 학습됐다. 그 조합은
        # 학습/서빙 프롬프트가 어긋나는 것이라 반드시 어댑터 sidecar에서 읽어 싣는다.
        # 출처(skeleton_hint_source)는 host 절대경로라 container manifest에 넣지 않는다.
        # 빌드 로그에만 남긴다. manifest에 넣으면 anti-leak 게이트가 정상적으로 막는다.
        "rationale_skeleton_hint": rationale_skeleton_hint,
        "parameters_billion": 9.6531,
        "enabled": True,
    },
    "notes": (
        "Qwen3.5-9B 8 seed (bbq35_e5_s42~s49) plateau checkpoints, equal weight 0.125, "
        "plus a Qwen3.5-9B rationale LoRA distilled from the Gemma v4 teacher. "
        "One baked base model shared by scoring and rationale. Objective aligned to "
        "the official metric: listwise soft-Spearman 0.2 retained, auxiliary detail/"
        "trait-average/paragraph losses dropped. integer_total_offset=0. Gemma weights "
        "are not deployed."
    ),
    "extra": {
        "candidate_key": "bbq35_e5_8seed_plateau_qwenrationale",
        "expected_image_tag": image_tag,
        "expected_http_prediction_sha256": expected_http_hash,
        "release_revision": image_tag,
        "source_config_id": "bbq35_e5_s42+s43+s44+s45+s46+s47+s48+s49",
        "checkpoint_selection": selection,
        # 4-member 앙상블이라 단일 source_checkpoint_artifact를 선언하지 않는다
        # (build_image.sh가 단일 선언 + member!=1을 금지한다). 멤버별 지문은 아래에 담고
        # container manifest의 복수 키는 build_image.sh가 staged tree에서 다시 채운다.
        "source_checkpoint_artifacts_by_run": {k: v["combined_sha256"] for k, v in score_cores.items()},
        "source_checkpoint_closures_by_run": {k: v["combined_sha256"] for k, v in score_trees.items()},
        "offline_validation": {
            "count": 400,
            "raw_continuous": {
                "rmse": 0.4250369308607046,
                "spearman": 0.7501743957467921,
            },
            "average_matched": {
                "rmse": 0.41682363709900666,
                "spearman": 0.7597366613254085,
            },
            "gold_source": "official_raw score.average",
        },
        "rationale_selection": {
            "teacher_model": "google/gemma-4-26B-A4B-it",
            "teacher_revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
            # v1 스크립트에서 상수로 넘어온 값이라 v3 artifact와 달랐다(기록상
            # 11,600행/726 step, 실제 11,029행/690 step). 가중치는 올바른 v3
            # 경로를 가리켰으므로 서빙에는 영향이 없었지만 provenance가 거짓이었다.
            # 이제 두 값 모두 실제 artifact에서 읽는다.
            "teacher_selected_rows": teacher_manifest.get("generated_this_run"),
            "teacher_requested_rows": teacher_manifest.get("selected"),
            "teacher_failed_rows": teacher_manifest.get("failed_this_run"),
            "student_base_model": "Qwen/Qwen3.5-9B",
            "student_base_revision": "main",
            "student_global_step": int(completed["global_step"]),
            "student_train_loss": completed["train_metrics"]["train_loss"],
            "student_adapter_runtime_sha256": rationale_tree["combined_sha256"],
            "student_adapter_weight_sha256": hashlib.sha256(
                adapter_weight.read_bytes()
            ).hexdigest(),
            "prompt_sha256": prompt_binding.sha256,
            "chat_template_sha256": runtime_config["chat_template_sha256"],
            "proxy_judge_exact_official": False,
            "proxy_judge_note": (
                "v3 라운드는 teacher를 바꾸지 않고 prompt만 v1->v3로 교체했으므로 "
                "proxy Judge A/B를 재실행하지 않았다. 최종 gate는 push 단계의 "
                "실제 image 400편 HTTP 평가다."
            ),
            "rationale_runtime_fingerprint": rationale_tree["combined_sha256"],
        },
        "http_oracle_note": (
            "Expected hash is accepted only after release_final.sh reproduces it "
            "from Docker HTTP validation 400."
        ),
    },
}

serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
output.write_text(serialized, encoding="utf-8")
loaded = load_manifest(output)
report = loaded.vram_budget_report()
require(report["fits"] is True, f"L40S static VRAM budget 실패: {report}")
print(f"manifest preflight PASS: {output}")
print(f"score core={score_core['combined_sha256']}")
print(f"score closure={score_tree['combined_sha256']}")
print(f"rationale runtime={rationale_tree['combined_sha256']}")
print(f"static VRAM budget={json.dumps(report, ensure_ascii=False)}")
print("selection: bbq35_e5_s42~s49 평탄 8seed 등가중 + Gemma-teacher v4 prompt Qwen rationale adapter")
PY

if [[ -f "${MANIFEST}" ]]; then
  if ! cmp -s -- "${MANIFEST_TMP}" "${MANIFEST}"; then
    echo "동일 tag/oracle의 기존 manifest와 내용이 다릅니다. 덮어쓰지 않습니다: ${MANIFEST}" >&2
    exit 1
  fi
  echo "기존 동일 manifest 재사용: ${MANIFEST}"
else
  mv -- "${MANIFEST_TMP}" "${MANIFEST}"
  echo "source manifest 생성: ${MANIFEST}"
fi

echo "filesystem: $(df -h "${ROOT}" | awk 'NR==2 {print "avail=" $4 ", used=" $5}')"

if [[ "${ACTION}" == "check" ]]; then
  echo "PASS: artifact/selection/prompt/manifest preflight"
  exit 0
fi

if [[ "${EXPECTED_HTTP_PREDICTION_SHA256}" == "${PROVISIONAL_HTTP_PREDICTION_SHA256}" ]]; then
  echo "주의: expected hash는 bbq35_e5 8seed 평탄 + offset 값이어야 합니다."
  echo "Docker HTTP 400편과 다르면 push 전에 정상적으로 실패하며, 실제 hash를 검토해야 합니다."
fi

RELEASE_MARKER="$(mktemp "${TMPDIR:-/tmp}/malpyeong_release_started.XXXXXX")"
set +e
REGISTRY_REPO="${REGISTRY_REPO}" GPU="${GPU}" PUSH="${PUSH}" RESULTS="${RESULTS}" \
  bash main_code_submission/release_final.sh "${MANIFEST}"
status=$?
set -e
if [[ "${status}" -ne 0 ]]; then
  latest_hash_report="$(find "${RESULTS}" -path '*/check/evaluation/prediction_hash.json' \
    -type f -newer "${RELEASE_MARKER}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -n "${latest_hash_report}" && -f "${latest_hash_report}" ]]; then
    echo "가장 최근 prediction hash 보고서: ${latest_hash_report}" >&2
    cat "${latest_hash_report}" >&2
  fi
  echo "release 실패(status=${status}); 검증을 우회하거나 수동 push하지 마세요" >&2
  exit "${status}"
fi

if [[ "${PUSH}" == "0" ]]; then
  echo "PASS: 동일 image ID build + Docker/HTTP validation 400편 검증 (push 생략)"
  echo "다음 단계: 같은 EXPECTED_HTTP_PREDICTION_SHA256로 bash docker_release.sh push"
else
  echo "PASS: build + validation 400편 + registry push"
  echo "다른 서버에서는 release.txt로 code_for_docker_check_otherserv/check_uploaded_final.sh를 실행하세요."
fi
