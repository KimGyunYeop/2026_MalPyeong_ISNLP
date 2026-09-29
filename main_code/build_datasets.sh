#!/usr/bin/env bash
set -euo pipefail

# Build every canonical dataset away from the fixed output directories.  Only
# a fully validated staging build is promoted, so training never sees a
# partially written JSONL file.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/datasets}"
RAW_ROOT="${RAW_ROOT:-${DATA_ROOT}/raw_dataset}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${RAW_ROOT}/official_competition}"
FORCE="${FORCE:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${SCRIPT_DIR}/../.venv-train/bin/python" ]]; then
  PYTHON_BIN="${SCRIPT_DIR}/../.venv-train/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ "${FORCE}" != "0" && "${FORCE}" != "1" ]]; then
  echo "FORCE must be 0 or 1: ${FORCE}" >&2
  exit 2
fi
if [[ ! -d "${RAW_ROOT}" || ! -d "${OFFICIAL_ROOT}" ]]; then
  echo "raw/official dataset directory is missing: ${RAW_ROOT}, ${OFFICIAL_ROOT}" >&2
  exit 2
fi

mkdir -p -- "${DATA_ROOT}"
LOCK_PATH="${DATA_ROOT}/.build_datasets.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "another dataset build is running: ${LOCK_PATH}" >&2
  exit 3
fi

OUTPUT_DIRECTORIES=(
  "processed_dataset"
  "processed_dataset_validation_leaked"
  "processed_dataset_only_official_competition"
  "processed_dataset_aihub_external_에세이"
  "processed_dataset_aihub_external_서술"
  "processed_dataset_aihub_external_논술"
  "processed_dataset_aihub_external_주제별"
)

is_nonempty_directory() {
  local directory="$1"
  [[ -d "${directory}" ]] && [[ -n "$(find "${directory}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

# Check every destination before spending time parsing the raw corpora.  A
# symlink is rejected even with FORCE because it makes the promotion target
# ambiguous.
for name in "${OUTPUT_DIRECTORIES[@]}"; do
  target="${DATA_ROOT}/${name}"
  if [[ -L "${target}" ]]; then
    echo "refusing symlink output target: ${target}" >&2
    exit 2
  fi
  if [[ -e "${target}" && ! -d "${target}" ]]; then
    if [[ "${FORCE}" != "1" ]]; then
      echo "output target is not a directory; use FORCE=1 to back it up: ${target}" >&2
      exit 2
    fi
  elif is_nonempty_directory "${target}" && [[ "${FORCE}" != "1" ]]; then
    echo "output is not empty; use FORCE=1 to create a timestamped backup: ${target}" >&2
    exit 2
  fi
done
if [[ -L "${DATA_ROOT}/schema.json" ]]; then
  echo "refusing symlink schema target: ${DATA_ROOT}/schema.json" >&2
  exit 2
fi
if [[ -e "${DATA_ROOT}/schema.json" && "${FORCE}" != "1" ]]; then
  echo "schema.json already exists; use FORCE=1 to create a timestamped backup" >&2
  exit 2
fi

STAGING_ROOT="$(mktemp -d "${DATA_ROOT}/.build_datasets.XXXXXX")"
PROMOTED=0
on_exit() {
  local status=$?
  if [[ "${PROMOTED}" == "1" ]]; then
    rmdir -- "${STAGING_ROOT}" 2>/dev/null || true
  elif [[ -d "${STAGING_ROOT}" ]]; then
    echo "build failed; staging output was kept for inspection: ${STAGING_ROOT}" >&2
  fi
  exit "${status}"
}
trap on_exit EXIT

echo "[BUILD] staging=${STAGING_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" \
  --raw-root "${RAW_ROOT}" \
  --official-root "${OFFICIAL_ROOT}" \
  --output-root "${STAGING_ROOT}" \
  --datasets all

echo "[VALIDATE] staging=${STAGING_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_data.py" \
  --data-root "${STAGING_ROOT}" \
  --official-root "${OFFICIAL_ROOT}"

# Promotion is same-filesystem rename.  Existing outputs are never deleted:
# FORCE=1 moves them to recoverable, timestamped backup paths first.
BACKUP_SUFFIX="$(date -u +%Y%m%dT%H%M%SZ).$$"
for name in "${OUTPUT_DIRECTORIES[@]}"; do
  source_path="${STAGING_ROOT}/${name}"
  target="${DATA_ROOT}/${name}"
  if [[ -e "${target}" ]]; then
    if [[ -d "${target}" ]] && ! is_nonempty_directory "${target}"; then
      rmdir -- "${target}"
    else
      backup="${target}.backup.${BACKUP_SUFFIX}"
      [[ ! -e "${backup}" ]] || { echo "backup already exists: ${backup}" >&2; exit 2; }
      mv -- "${target}" "${backup}"
      echo "[BACKUP] ${backup}"
    fi
  fi
  mv -- "${source_path}" "${target}"
done

if [[ -e "${DATA_ROOT}/schema.json" ]]; then
  schema_backup="${DATA_ROOT}/schema.json.backup.${BACKUP_SUFFIX}"
  [[ ! -e "${schema_backup}" ]] || { echo "backup already exists: ${schema_backup}" >&2; exit 2; }
  mv -- "${DATA_ROOT}/schema.json" "${schema_backup}"
  echo "[BACKUP] ${schema_backup}"
fi
mv -- "${STAGING_ROOT}/schema.json" "${DATA_ROOT}/schema.json"

PROMOTED=1
echo "[DONE] validated datasets promoted to ${DATA_ROOT}"
echo "       validate again: ${PYTHON_BIN} ${SCRIPT_DIR}/validate_data.py"
