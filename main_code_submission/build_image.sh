#!/usr/bin/env bash
# manifest를 읽어 제출 이미지를 만든다.
#
# 사용법:
#   bash main_code_submission/build_image.sh <manifest> [tag]
#   bash main_code_submission/build_image.sh \
#     main_code_submission/manifests/Y6_matched_fallback.json \
#     y6-cu128-offline-r6-20260811
#
# 오케스트레이터는 이미지 뒤에 인자를 붙이지 않으므로 여기서 모은 asset과 manifest가 이미지
# 안에서 그대로 쓰인다. 컨테이너 안의 manifest는 `/opt/submission`을 root로 다시 쓴다.
#
# Base weights는 항상 이미지에 넣는다. 시작 시 다운로드하는 mode는 제공하지 않는다.
set -euo pipefail

MANIFEST="${1:?manifest 경로가 필요합니다}"
TAG="${2:-v1}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${REPO_ROOT}/submission_assets"
IMAGE="${IMAGE_NAME:-malpyeong-writing-scorer}:${TAG}"

[[ -f "${MANIFEST}" ]] || {
  echo "manifest가 없습니다: ${MANIFEST}" >&2
  exit 2
}

# Release manifest가 tag를 선언하면 그대로 쓴다.
EXPECTED_IMAGE_TAG="$(python3 - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
extra = payload.get("extra") or {}
print(extra.get("expected_image_tag", ""))
PY
)"
if [[ -n "${EXPECTED_IMAGE_TAG}" && "${TAG}" != "${EXPECTED_IMAGE_TAG}" ]]; then
  echo "manifest 고정 tag와 build tag가 다릅니다: ${TAG} != ${EXPECTED_IMAGE_TAG}" >&2
  exit 2
fi

if [[ "${IMAGE_NAME:-}" == docker://* ]]; then
  echo "IMAGE_NAME에는 docker:// prefix를 넣지 마세요: ${IMAGE_NAME}" >&2
  exit 2
fi
IMAGE_REPOSITORY="${IMAGE_NAME:-malpyeong-writing-scorer}"
if [[ "${IMAGE_REPOSITORY##*/}" == *:* ]]; then
  echo "IMAGE_NAME에는 tag를 넣지 말고 두 번째 인자로 전달하세요: ${IMAGE_REPOSITORY}" >&2
  exit 2
fi

DOCKER_VIA_SG=0
init_docker() {
  command -v docker >/dev/null 2>&1 || {
    echo "docker가 없습니다. Docker Engine과 NVIDIA Container Toolkit을 설치하세요." >&2
    exit 1
  }
  if docker info >/dev/null 2>&1; then
    return
  fi
  if command -v sg >/dev/null 2>&1 \
      && sg docker -c 'docker info >/dev/null 2>&1'; then
    DOCKER_VIA_SG=1
    echo "Docker 보조 그룹이 현재 셸에 미반영되어 sg docker -c로 실행합니다."
    return
  fi
  echo "Docker daemon 접근 불가. 재로그인/newgrp docker 또는 docker group 설정을 확인하세요." >&2
  exit 1
}

docker_cmd() {
  if [[ "${DOCKER_VIA_SG}" == "0" ]]; then
    command docker "$@"
  else
    local command_line
    printf -v command_line '%q ' docker "$@"
    sg docker -c "${command_line}"
  fi
}

init_docker

cd "${REPO_ROOT}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/checkpoints" "${STAGE}/rationale_adapter" \
  "${STAGE}/hf/hub" "${STAGE}/hf_parts"

echo "== manifest에서 baked asset 수집 =="
PYTHONPATH="${REPO_ROOT}" python3 - "${MANIFEST}" "${STAGE}" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

from main_code_submission.artifact_integrity import (
    ArtifactIntegrityError,
    artifact_tree_fingerprint,
    checkpoint_fingerprint,
    require_fingerprint,
)
from main_code_submission.hf_cache_partition import (
    assemble_partitions,
    partition_cache,
    verify_partitions,
)
from main_code_submission.rationale_prompt_binding import (
    bind_adapter_prompt_to_manifest,
    prompt_binding_from_manifest,
)
from main_code.utils import model_cache_status

manifest_path, stage = Path(sys.argv[1]), Path(sys.argv[2])
raw = json.loads(manifest_path.read_text(encoding="utf-8"))

if raw.get("score_postprocess") != "average_matched":
    raise SystemExit(
        "submission manifest의 score_postprocess는 average_matched여야 합니다"
    )

CONTAINER_ROOT = "/opt/submission"

# Host provenance는 원본 manifest에 남기되, container manifest에는 image 밖의 절대경로나
# 이전 build의 배포 지문을 복사하지 않는다. 아래에서 실제 source/staged tree를 다시 읽어
# source/deployed 지문을 명시적으로 채운다.
extra = dict(raw.get("extra") or {})
declared_source_cores = extra.get("source_checkpoint_artifacts", {})
declared_source_closures = extra.get("source_checkpoint_closures", {})
declared_source_rationale = extra.get("source_rationale_runtime_artifact")
extra.pop("source_run", None)
declared_source_checkpoint = extra.pop("source_checkpoint_artifact", None)
legacy_source_checkpoint = extra.pop("checkpoint_artifact", None)
if declared_source_checkpoint is None:
    declared_source_checkpoint = legacy_source_checkpoint
for stale_key in (
    "source_checkpoint_artifacts",
    "source_checkpoint_closures",
    "deployed_checkpoint_artifacts",
    "deployed_checkpoint_closures",
    "source_rationale_runtime_artifact",
    "deployed_rationale_artifact",
    "artifact_integrity_required",
):
    extra.pop(stale_key, None)

staged = {
    "name": raw["name"],
    "root": CONTAINER_ROOT,
    "served_model_name": raw["served_model_name"],
    # Host에서 검증한 후처리 계약을 container manifest에도 그대로 보존한다.
    "score_postprocess": raw["score_postprocess"],
    "max_tokens": raw.get("max_tokens", 512),
    "seed": raw.get("seed", 42),
    # 표면을 반드시 옮긴다. 빠뜨리면 컨테이너가 기본값 official_raw로 돌아가고, Kiwi 문장
    # 경계 표면으로 학습한 checkpoint를 서빙할 때 train/serve 불일치가 조용히 생긴다.
    "essay_surface": raw.get("essay_surface", "official_raw"),
    # 정수 총점 offset도 반드시 옮긴다. 2026-08-22에 이 줄이 없어서 host manifest에
    # integer_total_offset=1이 적혀 있는데도 container는 기본값 0으로 돌아갔다.
    # 도커 HTTP 400편 검증에서 총점 평균이 10.16(=offset 0)으로 나와 잡혔다.
    "integer_total_offset": int(raw.get("integer_total_offset", 0) or 0),
    "score_members": [],
    "notes": raw.get("notes", ""),
    "extra": extra,
}

source_checkpoint_artifacts: dict[str, dict] = {}
source_checkpoint_closures: dict[str, dict] = {}
deployed_checkpoint_artifacts: dict[str, dict] = {}
deployed_checkpoint_closures: dict[str, dict] = {}

needed_models: dict[str, str] = {}   # model_id -> primary revision
required_model_revisions: dict[str, set[str]] = {}


def require_model(model_id: str, revision: str) -> None:
    """Record every requested revision; one model may be used by two adapters."""

    revision = revision or "main"
    needed_models.setdefault(model_id, revision)
    required_model_revisions.setdefault(model_id, set()).add(revision)


def stage_checkpoint(source: Path, target: Path) -> dict:
    """checkpoint를 복사하고 배포 환경에서 유효하지 않은 호스트 경로를 제거한다."""

    shutil.copytree(source, target, dirs_exist_ok=True)
    config_path = target / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # 구 checkpoint에는 학습 호스트의 절대 cache 경로가 저장돼 있다. 최신 loader는 Hugging
    # Face 기본 cache를 쓰므로 호환 key를 staging 산출물에서 제거한다.
    config.pop("model_cache_dir", None)
    # 2단계 AIHub run에는 phase-A adapter의 host 절대경로가 provenance로 남아 있다.
    # 배포 loader는 아래 checkpoint/adapter를 명시적으로 전달하므로 이 값은 사용하지 않으며,
    # image 안에 존재하지 않는 학습 호스트 경로를 보존하지 않는다.
    config["initial_lora_adapter"] = ""
    config["dataset_root"] = f"{CONTAINER_ROOT}/datasets_unused"
    config["extended_data_dir"] = f"{CONTAINER_ROOT}/datasets_unused"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_manifest.pop("model_cache_dir", None)
        checkpoint_manifest["initial_lora_adapter"] = ""
        manifest_path.write_text(
            json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    require_model(
        str(config["model_id"]),
        str(config.get("model_revision") or "main"),
    )
    return config


if declared_source_checkpoint is not None and len(raw["score_members"]) != 1:
    raise SystemExit(
        "단일 source_checkpoint_artifact는 score member가 하나일 때만 허용됩니다"
    )

for member in raw["score_members"]:
    source = Path(member["checkpoint"])
    if not source.is_dir():
        raise SystemExit(f"checkpoint가 없습니다: {source}")
    target = stage / "checkpoints" / member["name"]
    print(f"  checkpoint {member['name']}: {source}")
    try:
        source_artifact = checkpoint_fingerprint(source)
        source_closure = artifact_tree_fingerprint(source)
        if declared_source_cores:
            require_fingerprint(
                label=f"source checkpoint {member['name']}",
                expected=declared_source_cores.get(member["name"]),
                actual=source_artifact,
            )
        if declared_source_closures:
            require_fingerprint(
                label=f"source checkpoint closure {member['name']}",
                expected=declared_source_closures.get(member["name"]),
                actual=source_closure,
            )
        if declared_source_checkpoint is not None:
            require_fingerprint(
                label=f"source checkpoint {member['name']}",
                expected=declared_source_checkpoint,
                actual=source_artifact,
            )
    except ArtifactIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    source_checkpoint_artifacts[member["name"]] = source_artifact
    source_checkpoint_closures[member["name"]] = source_closure
    print(
        f"    source core={source_artifact['combined_sha256']} "
        f"closure={source_closure['combined_sha256']}"
    )
    stage_checkpoint(source, target)
    try:
        # config/manifest의 host path scrub가 끝난 뒤 실제 배포 파일을 다시 hash한다. 이
        # top-level submission manifest는 checkpoint tree 밖에 있어 self-reference가 없다.
        deployed_checkpoint_artifacts[member["name"]] = checkpoint_fingerprint(
            target
        )
        deployed_checkpoint_closures[member["name"]] = artifact_tree_fingerprint(
            target
        )
    except ArtifactIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "    deployed "
        f"core={deployed_checkpoint_artifacts[member['name']]['combined_sha256']} "
        f"closure={deployed_checkpoint_closures[member['name']]['combined_sha256']}"
    )
    staged["score_members"].append(
        {**member, "checkpoint": f"checkpoints/{member['name']}"}
    )

rationale = dict(raw.get("rationale", {}))
source_rationale_runtime_artifact = None
deployed_rationale_artifact = None
if rationale.get("adapter"):
    source = Path(rationale["adapter"])
    if not source.is_dir():
        raise SystemExit(f"근거 어댑터가 없습니다: {source}")
    if declared_source_rationale is not None:
        require_fingerprint(
            label="source rationale adapter",
            expected=declared_source_rationale,
            actual=artifact_tree_fingerprint(source),
        )
    print(f"  근거 어댑터: {source}")
    try:
        prompt_binding = bind_adapter_prompt_to_manifest(rationale, source)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "    rationale prompt="
        f"{prompt_binding.prompt_id}@{prompt_binding.sha256}"
    )
    target = stage / "rationale_adapter"
    try:
        # transformers Trainer의 pickle에는 학습 output_dir 같은 host 절대경로가 남는다.
        # 서빙에 쓰이지 않으므로 source runtime 지문과 image 양쪽에서 제외한다.
        source_rationale_runtime_artifact = artifact_tree_fingerprint(
            source, excluded_relative_paths={"training_args.bin"}
        )
    except ArtifactIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("training_args.bin"),
    )
    leaked_training_args = sorted(target.rglob("training_args.bin"))
    if leaked_training_args:
        raise SystemExit(
            f"rationale training_args.bin이 staging에 남았습니다: {leaked_training_args}"
        )
    try:
        deployed_rationale_artifact = artifact_tree_fingerprint(target)
        require_fingerprint(
            label="rationale runtime artifact copy",
            expected=source_rationale_runtime_artifact,
            actual=deployed_rationale_artifact,
        )
    except ArtifactIntegrityError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "    rationale runtime artifact="
        f"{deployed_rationale_artifact['combined_sha256']} "
        "(training_args.bin 제외)"
    )
    rationale["adapter"] = "rationale_adapter"
else:
    # Disabled diagnostics and legacy manifests also receive an explicit prompt
    # binding in the staged manifest.  The all-missing legacy rule is baseline.
    try:
        prompt_binding = prompt_binding_from_manifest(rationale)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rationale.update(prompt_binding.manifest_fields())
if rationale.get("enabled", True) and rationale.get("base_model"):
    require_model(
        str(rationale["base_model"]),
        str(rationale.get("base_model_revision") or "main"),
    )
staged["rationale"] = rationale

staged["extra"].update(
    {
        "artifact_integrity_required": True,
        "source_checkpoint_artifacts": source_checkpoint_artifacts,
        "source_checkpoint_closures": source_checkpoint_closures,
        "deployed_checkpoint_artifacts": deployed_checkpoint_artifacts,
        "deployed_checkpoint_closures": deployed_checkpoint_closures,
    }
)
if source_rationale_runtime_artifact is not None:
    staged["extra"][
        "source_rationale_runtime_artifact"
    ] = source_rationale_runtime_artifact
    staged["extra"]["deployed_rationale_artifact"] = deployed_rationale_artifact


def host_absolute_paths(value, location: str = "$") -> list[str]:
    """Container root 밖의 절대 host path가 staged manifest에 남았는지 찾는다."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(host_absolute_paths(item, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(host_absolute_paths(item, f"{location}[{index}]"))
    elif isinstance(value, str) and value.startswith("/"):
        if value != CONTAINER_ROOT and not value.startswith(f"{CONTAINER_ROOT}/"):
            found.append(f"{location}={value}")
    return found


# 회귀 방지 게이트. 후처리 계약 필드가 조용히 떨어지면 여기서 죽는다.
for _k, _default in (("score_postprocess", None), ("essay_surface", "official_raw"),
                     ("integer_total_offset", 0)):
    _src = raw.get(_k, _default)
    if _k == "integer_total_offset":
        _src = int(_src or 0)
    if staged.get(_k) != _src:
        raise SystemExit(
            f"container manifest에서 후처리 계약 필드가 유실됐습니다: {_k} "
            f"source={_src!r} staged={staged.get(_k)!r}"
        )

leaked_host_paths = host_absolute_paths(staged)
if leaked_host_paths:
    raise SystemExit(
        "container manifest에 host 절대경로가 남았습니다:\n  "
        + "\n  ".join(leaked_host_paths)
    )

(stage / "submission_manifest.json").write_text(
    json.dumps(staged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(f"  manifest: {stage / 'submission_manifest.json'}")
print(f"  필요한 base 모델: {json.dumps(needed_models, ensure_ascii=False)}")
(stage / "needed_models.json").write_text(
    json.dumps(
        {key: sorted(value) for key, value in required_model_revisions.items()},
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

target_hub = stage / "hf" / "hub"
parts_root = stage / "hf_parts"
partition_manifest_path = stage / "hf_partition_manifest.json"

# --- base 가중치 staging -------------------------------------------------------
# HF 캐시 layout을 그대로 유지한다. snapshot 안의 파일은 `../../blobs/<sha>` 형태의 **상대**
# symlink이므로, 필요한 blob만 골라 복사하고 symlink를 다시 만들면 오프라인 캐시가 완성된다.
# 디렉토리를 통째로 복사하면 예전 revision의 blob까지 들어가 이미지가 불필요하게 커진다.
host_hub = Path.home() / ".cache" / "huggingface" / "hub"
if not host_hub.is_dir():
    raise SystemExit(f"Hugging Face 기본 캐시를 찾을 수 없습니다: {host_hub}")
print(f"  HF 캐시: {host_hub}")

target_hub.mkdir(parents=True, exist_ok=True)
total = 0
for model_id, _primary_revision in needed_models.items():
    repo_dir = host_hub / ("models--" + model_id.replace("/", "--"))
    if not repo_dir.is_dir():
        raise SystemExit(
            f"{model_id}의 캐시가 없습니다: {repo_dir}\n"
            f"  먼저 내려받으세요: huggingface-cli download {model_id}"
        )
    resolved: dict[str, str] = {}
    for required_revision in sorted(required_model_revisions[model_id]):
        complete, detail = model_cache_status(host_hub, model_id, required_revision)
        if not complete:
            raise SystemExit(
                f"{model_id}@{required_revision}: 불완전한 host cache라 baked image를 "
                f"만들 수 없습니다 ({detail})"
            )
        print(f"    원본 검사: {detail}")
        required_ref = repo_dir / "refs" / required_revision
        resolved[required_revision] = (
            required_ref.read_text(encoding="utf-8").strip()
            if required_ref.is_file()
            else required_revision
        )
    if len(set(resolved.values())) != 1:
        raise SystemExit(
            f"{model_id}: score/rationale가 서로 다른 base revision을 요구합니다: {resolved}"
        )
    commit = next(iter(resolved.values()))
    snapshot = repo_dir / "snapshots" / commit
    if not snapshot.is_dir():
        candidates = sorted((repo_dir / "snapshots").iterdir())
        if len(candidates) != 1:
            raise SystemExit(f"{model_id}: snapshot을 특정할 수 없습니다 {candidates}")
        snapshot = candidates[0]
        commit = snapshot.name

    out_repo = target_hub / repo_dir.name
    (out_repo / "snapshots" / commit).mkdir(parents=True, exist_ok=True)
    (out_repo / "blobs").mkdir(parents=True, exist_ok=True)
    (out_repo / "refs").mkdir(parents=True, exist_ok=True)
    for required_revision in required_model_revisions[model_id]:
        # Commit hashes resolve directly through snapshots/<hash>. Named refs need
        # an explicit cache ref inside the offline image.
        if required_revision != commit:
            ref_target = out_repo / "refs" / required_revision
            ref_target.parent.mkdir(parents=True, exist_ok=True)
            # huggingface_hub 1.27 treats the ref file literally and does not
            # strip a trailing newline when constructing snapshots/<commit>.
            ref_target.write_text(commit, encoding="utf-8")
            stored_ref = ref_target.read_text(encoding="utf-8")
            if stored_ref != commit or any(char.isspace() for char in stored_ref):
                raise SystemExit(
                    f"{model_id}@{required_revision}: staging ref에 공백/줄바꿈이 "
                    f"들어갔습니다: {stored_ref!r}"
                )

    copied = 0
    for entry in sorted(snapshot.rglob("*")):
        if entry.is_dir():
            continue
        relative = entry.relative_to(snapshot)
        destination = out_repo / "snapshots" / commit / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry.is_symlink():
            blob = Path(os.readlink(entry)).name
            blob_source = repo_dir / "blobs" / blob
            blob_target = out_repo / "blobs" / blob
            if not blob_target.exists():
                shutil.copy2(blob_source, blob_target)
                copied += blob_target.stat().st_size
            # snapshot -> blobs 상대 경로를 원본과 같은 깊이로 다시 만든다.
            depth = len(relative.parts) - 1
            prefix = "../" * (depth + 2)
            destination.symlink_to(f"{prefix}blobs/{blob}")
        else:
            shutil.copy2(entry, destination)
            copied += destination.stat().st_size
    for required_revision in sorted(required_model_revisions[model_id]):
        staged_complete, staged_detail = model_cache_status(
            target_hub, model_id, required_revision
        )
        if not staged_complete:
            raise SystemExit(
                f"{model_id}@{required_revision}: staging cache 완전성 검사 실패 "
                f"({staged_detail})"
            )
    total += copied
    print(
        f"    {model_id} @ {commit[:12]}: {copied / 2**30:.2f} GiB, "
        f"staging 검사 완료"
    )
print(f"  base 가중치 합계: {total / 2**30:.2f} GiB")

# --- registry-portable layer partition ---------------------------------------
# GHCR은 한 layer가 10GB를 넘으면 거부한다. 각 snapshot symlink를 대상 blob과 같은
# partition에 넣고, Dockerfile의 고정 8개 COPY가 각각 8GiB 미만이 되게 한다.
partition_manifest = partition_cache(
    target_hub, parts_root, partition_manifest_path
)
for part in partition_manifest["parts"]:
    print(
        f"    {part['name']}: files={part['entry_count']}, "
        f"payload={part['file_bytes'] / 2**30:.2f} GiB, "
        f"layer-estimate={part['estimated_layer_bytes'] / 2**30:.2f} GiB"
    )

# 반복 COPY가 만든 최종 cache tree를 hardlink 기반으로 재현한 뒤 기존 completeness gate를
# 다시 실행한다. partition별 source가 맞더라도 합쳤을 때 refs/snapshot/blob가 어긋나면
# Docker build 전에 중단한다.
verification_root = stage / "hf_partition_assembly_check"
verification_hub = verification_root / "hub"
assemble_partitions(parts_root, verification_hub, partition_manifest)
for model_id in sorted(required_model_revisions):
    for required_revision in sorted(required_model_revisions[model_id]):
        assembled_complete, assembled_detail = model_cache_status(
            verification_hub, model_id, required_revision
        )
        if not assembled_complete:
            raise SystemExit(
                f"{model_id}@{required_revision}: partition 재조립 cache 검사 실패 "
                f"({assembled_detail})"
            )
        print(f"    partition 재조립 검사: {assembled_detail}")
shutil.rmtree(verification_root)
PY

echo "== staged asset 크기 =="
du -sh --apparent-size "${STAGE}" | sed 's/^/   /'

echo "== docker build =="
docker_cmd build -t "${IMAGE}" -f main_code_submission/Dockerfile .

echo "== 최종 image layer 크기 검사 (GHCR <10GB, HF part <8GiB) =="
IMAGE_HISTORY="${STAGE}/image_history.jsonl"
docker_cmd history --no-trunc --format '{{json .}}' "${IMAGE}" > "${IMAGE_HISTORY}"
PYTHONPATH="${REPO_ROOT}" python3 -m main_code_submission.hf_cache_partition \
  audit-history \
  --history "${IMAGE_HISTORY}" \
  --manifest "${STAGE}/hf_partition_manifest.json" \
  --output "${STAGE}/image_layer_audit.json"

echo "== 이미지 크기 =="
docker_cmd image inspect "${IMAGE}" --format '   {{.Size}} bytes' \
  | awk '{printf "   %.2f GiB\n", $1/1073741824}'

cat <<EOF

완료: ${IMAGE}

이 스크립트는 image build만 한다. 수동 docker push는 검증을 우회하므로
최종 제출에 사용하지 말고, 새 canonical tag에서 아래 단일 release를 실행한다.

  REGISTRY_REPO=OWNER/REPOSITORY GPU=all PUSH=1 \
    bash main_code_submission/release_final.sh ${MANIFEST}

release_final.sh는 새로 build한 동일 image ID의 no-argument contract·validation 400편을
모두 통과한 경우에만 새 registry tag로 push한다. 현재 확정 c02+Gemma-student 후보는
저장소 루트의 docker_release.sh가 artifact와 manifest를 먼저 고정한 뒤 이 경로를 호출한다.
EOF
