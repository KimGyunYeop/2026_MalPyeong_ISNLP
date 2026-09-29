#!/usr/bin/env bash
# 기술서 ABCD 재학습 산출물 검사. 과거 AB 체크포인트/성능/해시를 가져오지 않는다.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
ACTION="${1:-check}"
case "${ACTION}" in check|verify|push) ;; *) echo '사용법: bash docker_release.sh check|verify|push' >&2; exit 2 ;; esac
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/report_abcd}"
DEFAULT_SCORES=""
for seed in 42 43 44 45 46 47 48 49; do
  DEFAULT_SCORES+="${DEFAULT_SCORES:+,}${OUT_ROOT}/score_s${seed}/checkpoint"
done
SCORE_CHECKPOINTS="${SCORE_CHECKPOINTS:-${DEFAULT_SCORES}}"
RATIONALE_ADAPTER="${RATIONALE_ADAPTER:-${OUT_ROOT}/rationale/final_adapter}"
IMAGE_TAG="${IMAGE_TAG:-report-abcd-qwen35-8seed-r1}"
MANIFEST_DIR="${MANIFEST_DIR:-${ROOT}/main_code_submission/results/report_manifests}"
EXPECTED_HTTP_PREDICTION_SHA256="${EXPECTED_HTTP_PREDICTION_SHA256:-}"
REGISTRY_REPO="${REGISTRY_REPO:-gyunyeop/writing-scorer}"
GPU="${GPU:-0}"
if [[ "${ACTION}" != check && ! "${EXPECTED_HTTP_PREDICTION_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo '새 ABCD 모델의 400편 예측 해시를 EXPECTED_HTTP_PREDICTION_SHA256으로 지정하세요.' >&2
  echo '최초 측정: check로 manifest 생성 → build_image.sh → evaluate_http.py.' >&2
  exit 2
fi
exec 9>"${TMPDIR:-/tmp}/malpyeong_docker_release.lock"
flock -n 9 || { echo '다른 Docker build가 실행 중입니다' >&2; exit 1; }
MANIFEST="${MANIFEST_DIR}/${IMAGE_TAG}_${EXPECTED_HTTP_PREDICTION_SHA256:0:12}.json"
"${PYTHON_BIN}" - "${ROOT}" "${SCORE_CHECKPOINTS}" "${RATIONALE_ADAPTER}" \
  "${IMAGE_TAG}" "${EXPECTED_HTTP_PREDICTION_SHA256}" "${MANIFEST}" <<'PY'
import dataclasses, json, math, re, sys
from pathlib import Path
from main_code.config import load_config as load_score
from main_code_relonation.config import load_config as load_rationale
from main_code_submission.artifact_integrity import checkpoint_fingerprint, artifact_tree_fingerprint
from main_code_submission.config import load_manifest
from main_code_submission.rationale_prompt_binding import prompt_binding_from_adapter
root = Path(sys.argv[1])
paths = [Path(p).expanduser().resolve() for p in sys.argv[2].split(',') if p.strip()]
rationale = Path(sys.argv[3]).expanduser().resolve()
tag, oracle, output_arg = sys.argv[4:]
output = Path(output_arg)
def require(ok, message):
    if not ok:
        raise SystemExit(message)
def read(path):
    require(path.is_file(), f'필수 artifact가 없습니다: {path}')
    return json.loads(path.read_text())
require(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}', tag), '잘못된 image tag')
require(not oracle or re.fullmatch(r'[0-9a-f]{64}', oracle), '잘못된 prediction hash')
require(len(paths) == 8 and len(set(paths)) == 8, '서로 다른 8개 checkpoint가 필요합니다')
expected = dataclasses.asdict(load_score(root/'main_code/configs/report_abcd.json'))
members, cores, closures, selections, seeds = [], {}, {}, {}, []
for path in paths:
    cfg = read(path/'config.json')
    actual = dataclasses.asdict(load_score(path/'config.json'))
    diff = {k:(v,actual.get(k)) for k,v in expected.items()
            if k not in {'seed','dataset_root','extended_data_dir'} and v != actual.get(k)}
    require(not diff, f'기술서 ABCD 설정과 다릅니다: {path}: {diff}')
    seed = cfg['seed']; seeds.append(seed)
    state = read(path.parent/'trainer/trainer_state.json')
    run = read(path.parent/'run.json')
    require(not (path/'selection.json').exists(), f'최종 step checkpoint가 아닙니다: {path}')
    require(state['global_step'] == state['max_steps'] == 1104, f'1104 step 미완료: {path}')
    require(run['training_rows'] == 11600, f'학습 11600편이 아닙니다: {path}')
    name = f'report_abcd_s{seed}'
    members.append(dict(name=name, checkpoint=str(path), backbone_key='qwen35_9b', parameters_billion=9.6531, weight=0.125))
    cores[name] = checkpoint_fingerprint(path)
    closures[name] = artifact_tree_fingerprint(path)
    selections[name] = {'policy':'final_checkpoint_no_selection','global_step':1104}
require(sorted(seeds) == list(range(42,50)), f'seeds must be 42..49: {seeds}')
recipe = load_rationale(root/'main_code_relonation/recipes/r18_report_qwen35.json')
resolved = load_rationale(rationale.parent/'resolved_config.json')
left, right = recipe.to_dict(), resolved.to_dict()
diff = {k:(v,right.get(k)) for k,v in left.items() if k != 'rationale_prompt_source' and v != right.get(k)}
require(not diff, f'근거 recipe 불일치: {diff}')
run, completed = read(rationale.parent/'run_manifest.json'), read(rationale.parent/'completed.json')
require(run['training_row_count'] == 11600, '근거 모델은 accepted 11600편 학습이 필요합니다')
require(completed['status'] == 'complete', '근거 학습 미완료')
steps = math.ceil(11600 / (recipe.batch_size * recipe.gradient_accumulation)) * recipe.epochs
require(completed['global_step'] == steps, f'근거 학습 step 불일치: {completed["global_step"]} != {steps}')
binding = prompt_binding_from_adapter(rationale)
require(binding.sha256 == recipe.rationale_prompt_sha256, '근거 prompt 불일치')
runtime, adapter = read(rationale/'rationale_runtime_config.json'), read(rationale/'adapter_config.json')
require(adapter['base_model_name_or_path'] == recipe.model_id, '근거 base model 불일치')
require(adapter['r'] == recipe.lora_rank and adapter['lora_alpha'] == recipe.lora_alpha, '근거 LoRA 불일치')
require(set(adapter['target_modules']) == set(recipe.lora_targets), '근거 LoRA targets 불일치')
require(runtime['score_mode'] == 'fixed', '근거 점수 고정 계약 불일치')
require(runtime['training_conditioning_score_postprocess'] == 'average_matched', '근거 정수 점수 조건 불일치')
manifest = {
    'name':'report_abcd_qwen35_8seed', 'root':str(root), 'served_model_name':'malpyeong-writing-scorer',
    'score_postprocess':'average_matched', 'integer_total_offset':0, 'essay_surface':'official_raw',
    'max_tokens':512, 'temperature':0.0, 'top_p':1.0, 'seed':42, 'score_members':members,
    'rationale':{
        'base_model':recipe.model_id, 'base_model_revision':recipe.model_revision,
        'adapter':str(rationale), 'share_backbone_key':'qwen35_9b', 'load_in_4bit':False,
        'max_new_tokens':recipe.max_new_tokens, 'max_length':recipe.max_length, 'deadline_seconds':100.0,
        'chat_template_kwargs':recipe.chat_template_kwargs, 'chat_template_sha256':runtime['chat_template_sha256'],
        'rationale_prompt_id':binding.prompt_id, 'rationale_prompt_text':binding.text,
        'rationale_prompt_sha256':binding.sha256, 'rationale_skeleton_hint':recipe.rationale_skeleton_hint,
        'parameters_billion':9.6531, 'enabled':True,
    },
    'notes':'Report ABCD retraining artifacts. No historical AB metrics or prediction oracle inherited.',
    'extra':{
        'expected_image_tag':tag, 'expected_http_prediction_sha256':oracle or None,
        'checkpoint_selection':selections, 'source_checkpoint_artifacts':cores, 'source_checkpoint_closures':closures,
        'source_rationale_runtime_artifact':artifact_tree_fingerprint(rationale),
        'rationale_training_rows':run['training_row_count'], 'rationale_steps':completed['global_step'],
        'verified_metrics':None, 'evaluation_status':'deferred_by_user',
        'report_differences':['rationale: 726 steps / 37 warmup, PDF: 725 / 36',
                              'score: 56 warmup, PDF: 55',
                              'prompt_8 versus repository v4 identity unverified',
                              'auxiliary-loss equations and sampled retry differ; see audit report'],
    },
}
serialized = json.dumps(manifest, ensure_ascii=False, indent=2)+'\n'
output.parent.mkdir(parents=True, exist_ok=True)
require(not output.exists() or output.read_text() == serialized, f'기존 manifest를 덮어쓰지 않습니다: {output}')
output.write_text(serialized)
config = load_manifest(output)
require(config.vram_budget_report()['fits'], '정적 VRAM 예산 초과')
print(f'PASS: ABCD artifact/config preflight: {output}')
print('실제 성능과 PDF 완전 일치는 아직 검증되지 않았습니다.')
PY
[[ "${ACTION}" != check ]] || exit 0
PUSH=0
[[ "${ACTION}" != push ]] || PUSH=1
REGISTRY_REPO="${REGISTRY_REPO}" GPU="${GPU}" PUSH="${PUSH}" \
  bash main_code_submission/release_final.sh "${MANIFEST}"
