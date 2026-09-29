# 2026 AI 말평 — 글쓰기 채점 능력 평가

## 최종 방법론

Qwen3.5-9B를 공유하는 **채점 LoRA 8개와 근거 생성 LoRA 1개**로 내용·구성·표현 점수와 근거를 생성한다. 백본은 BF16으로 사용하며, revision은 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`로 고정한다.

채점 모델은 다음 네 요소를 함께 학습한다(ABCD).

- **A — 분포 기반 채점:** 영역별 5등급 확률분포의 기댓값을 연속 점수로 사용하고 MSE와 CE로 학습한다.
- **B — 순위 학습:** 영역별 Soft-Spearman 손실을 추가한다. 가중치는 0.2, 온도는 0.5이다.
- **C — 평가자 준거 보조 학습:** `rater_set` head에 expected MSE와 rater CE를 각각 0.25의 가중치로 적용한다. 최종 영역 점수는 직접 예측 head에서 얻는다.
- **D — 문단 표현:** 구성 영역에 `paragraph_mean` pooling을 적용한다.

시드 42~49의 최종 체크포인트에서 얻은 연속 점수를 각각 0.125로 평균하고, **SMR(`average_matched`)을 한 번 적용**하여 1~5의 정수 점수로 변환한다. 점수 offset은 0이다.

근거 모델은 Gemma4-26B-A4B-it AWQ teacher가 고정 점수를 조건으로 생성한 데이터로 학습한다. 추론에서는 채점 결과를 고정하고 근거만 생성한다. 기본 생성은 temperature 0 / top_p 1, 최대 2,048 tokens이며 실패 시 재생성을 수행한다. 근거 학습과 추론의 프롬프트는 레시피에 지정된 [rationale_prompt_v4.txt](main_code_relonation/prompts/rationale_prompt_v4.txt)를 사용한다.

| 설정 | 채점 | 근거 생성 |
| --- | --- | --- |
| 레시피 | [report_abcd.json](main_code/configs/report_abcd.json) | [r18_report_qwen35.json](main_code_relonation/recipes/r18_report_qwen35.json) |
| 학습 데이터 | 11,600편 | 조건·형식 검사를 통과한 11,600편 |
| 학습량 | 시드별 1,104 steps, 64 steps마다 검증 | 2 epochs |
| LoRA | r32 / alpha64 / dropout 0.05, attention + MLP | r32 / alpha64 / dropout 0.05, q/k/v/o |
| 배치 × gradient accumulation | 32 × 1 | 2 × 16 |
| 학습률 | LoRA 4e-5, 출력층 2e-4 | 4e-5 |
| 최대 입력 길이 | 4,096 tokens | 8,192 tokens |
| warmup 비율 | 0.05 | 0.05 |
| 시드 | 42~49 | 42 |

## Usage

### 1. 환경 생성

Linux, Conda, C/C++ 빌드 도구, CUDA 13.0을 지원하는 NVIDIA 드라이버를 준비한다. Docker 제출에는 Docker Engine과 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)이 필요하다. 저장소를 받은 뒤 나머지 명령은 저장소 루트에서 순서대로 실행한다.

```bash
# 명령이 실패하면 다음 단계로 진행하지 않는다.
set -e

git clone https://github.com/KimGyunYeop/2026_MalPyeong_ISNLP.git
cd 2026_MalPyeong_ISNLP

conda create -p .venv-train python=3.13.9 pip -y
conda activate "$PWD/.venv-train"
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install \
  -r requirements-train.txt -r requirements-eval.txt \
  -r main_code_relonation/requirements.txt \
  transformers==5.14.0 peft==0.19.1 accelerate==1.14.0 \
  safetensors==0.8.0 bitsandbytes==0.49.2

export PYTHON_BIN="$PWD/.venv-train/bin/python"
export GPU=0
export DATASET_ROOT="$PWD/main_code/datasets"
export OUT_ROOT="$PWD/results/report_abcd"
```

학습과 Docker 빌드에 사용할 백본을 내려받는다.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
)
PY
```

### 2. 데이터 준비

공식 train·validation JSONL과 국립국어원 원천 자료를 다음 위치에 둔다. `NIKL_GRADING WRITING DATA`로 시작하는 원본 폴더 3개의 이름을 유지하고 각 폴더 바로 아래에 JSON 파일을 둔다.

```text
main_code/datasets/raw_dataset/
├── official_competition/
│   ├── <공식 파일명>train.jsonl
│   └── <공식 파일명>validation.jsonl
├── NIKL_GRADING WRITING DATA.../
├── NIKL_GRADING WRITING DATA.../
└── NIKL_GRADING WRITING DATA.../
```

```bash
python -m main_code.prepare_data \
  --raw-root "$DATASET_ROOT/raw_dataset" \
  --official-root "$DATASET_ROOT/raw_dataset/official_competition" \
  --output-root "$DATASET_ROOT" \
  --datasets competition
```

`processed_dataset/train.jsonl`에 11,600편, `validation.jsonl`에 400편이 생성된다. 채점의 준거 보조 학습에는 원천 평가자 라벨이 필요하다. Docker 검증용 `DATA_FILE`은 전처리된 파일 대신 **공식 validation 원본**의 절대 경로로 지정한다.

```bash
export DATA_FILE="/absolute/path/to/official_validation.jsonl"
```

### 3. 채점 모델 학습

```bash
for seed in 42 43 44 45 46 47 48 49; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m main_code.train \
    --config main_code/configs/report_abcd.json \
    --dataset-root "$DATASET_ROOT" \
    --seed "$seed" \
    --output-dir "$OUT_ROOT/score_s${seed}"
done
```

최종 어댑터와 출력층은 `results/report_abcd/score_s42/checkpoint`부터 `score_s49/checkpoint`까지 저장된다. 새로운 학습에는 비어 있는 출력 경로를 사용한다.

### 4. 근거 학습 데이터 생성

같은 프롬프트와 고정 점수 조건으로 생성한 학습 가능 데이터 11,600편이 있으면 이 단계를 건너뛰고 다음 절의 `PSEUDO_FILE`에 지정한다.

teacher용 vLLM은 별도 환경에 설치한다.

```bash
conda create -p .venv-vllm python=3.12.13 pip -y
.venv-vllm/bin/python -m pip install vllm==0.25.1 transformers==5.14.0 ninja

export VLLM_PYTHON="$PWD/.venv-vllm/bin/python"
export TEACHER_MODEL="/path/to/Gemma4-26B-A4B-it-AWQ"
export TEACHER_MODEL_REVISION="<teacher의 고정 commit>"
export RATIONALE_PROMPT_FILE="$PWD/main_code_relonation/prompts/rationale_prompt_v4.txt"
export TEACHER_OUT_ROOT="$PWD/main_code_relonation/results"
export RUN_NAME="report_rationale_v4"
```

`TEACHER_MODEL`에는 준비한 AWQ 모델의 로컬 경로나 Hugging Face 저장소를, `TEACHER_MODEL_REVISION`에는 해당 모델의 revision을 넣는다. teacher 서버를 실행하고 준비될 때까지 기다린다. 스크립트의 기본 여유 VRAM 검사는 80 GiB이며, 사용하는 AWQ 모델과 GPU에 맞춰 `TEACHER_MIN_FREE_GIB`와 `TEACHER_GPU_UTIL`을 지정할 수 있다.

```bash
STAGE=teacher OUT_ROOT="$TEACHER_OUT_ROOT" \
  bash main_code_relonation/run_rationale_pipeline.sh
```

다른 터미널에서 저장소 루트로 이동해 학습 환경을 활성화하고 위 환경변수를 동일하게 설정한 뒤 실행한다. 인간 점수에 SMR을 적용한 정수 점수를 조건으로 근거를 생성한다.

```bash
STAGE=generate OUT_ROOT="$TEACHER_OUT_ROOT" \
  TRAIN_INPUT="$DATASET_ROOT/processed_dataset/train.jsonl" \
  SCORE_SOURCE=human_average_matched MIN_TEACHER_SUCCESS_RATE=1.0 \
  bash main_code_relonation/run_rationale_pipeline.sh
```

출력은 `main_code_relonation/results/report_rationale_v4/pseudo_train.jsonl`이다. 실패한 항목이 있으면 같은 명령을 재실행한다. 같은 GPU에서 근거 모델을 학습하려면 생성 완료 후 teacher 서버를 종료한다.

### 5. 근거 모델 학습

```bash
export PSEUDO_FILE="$PWD/main_code_relonation/results/report_rationale_v4/pseudo_train.jsonl"

python - <<'PY'
import os
from pathlib import Path
from main_code_relonation.artifacts import read_rows
from main_code_relonation.config import load_config
from main_code_relonation.train import (
    accepted_rows, validate_training_prompt_rows, validate_training_score_rows,
)

config = load_config("main_code_relonation/recipes/r18_report_qwen35.json")
rows = accepted_rows(read_rows(Path(os.environ["PSEUDO_FILE"])))
assert len(rows) == 11600, f"학습 가능한 데이터: {len(rows)}/11600"
validate_training_prompt_rows(rows, config)
validate_training_score_rows(rows, config)
print("PASS: 근거 학습 데이터 11600편, 프롬프트 및 고정 점수 검사")
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m main_code_relonation.train \
  --recipe main_code_relonation/recipes/r18_report_qwen35.json \
  --train-file "$PSEUDO_FILE" \
  --output-dir "$OUT_ROOT/rationale"
```

검사를 통과한 뒤 학습한다. 최종 어댑터는 `results/report_abcd/rationale/final_adapter`에 저장된다.

### 6. Docker 이미지 빌드 및 검증

공개 Docker Hub 저장소와 아직 사용하지 않은 태그를 지정한다. 매니페스트 생성 시 채점 체크포인트 8개와 근거 어댑터의 설정·학습 완료 여부·파일 해시를 검사한다.

```bash
export REGISTRY_REPO="YOUR_DOCKERHUB_ACCOUNT/writing-scorer"
export IMAGE_TAG="report-abcd-qwen35-8seed-r1"
export MANIFEST_DIR="$PWD/main_code_submission/results/report_manifests"
unset EXPECTED_HTTP_PREDICTION_SHA256

bash docker_release.sh check
MANIFEST="$MANIFEST_DIR/${IMAGE_TAG}_.json"
bash main_code_submission/build_image.sh "$MANIFEST" "$IMAGE_TAG"
```

이미지에는 백본·어댑터·프롬프트가 포함되며, 추가 실행 인자 없이 포트 8000에서 API를 제공한다. 컨테이너를 실행하여 공식 validation 400편을 검사한다.

```bash
docker run --rm -d --name malpyeong-score-check \
  --gpus "device=$GPU" -p 127.0.0.1:8000:8000 \
  "malpyeong-writing-scorer:$IMAGE_TAG"

python code_for_docker_check_otherserv/evaluate_http.py \
  --base-url http://127.0.0.1:8000 \
  --input "$DATA_FILE" \
  --output-dir "$OUT_ROOT/http_validation" \
  --expected-count 400 \
  --expected-model malpyeong-writing-scorer

docker stop malpyeong-score-check
```

검사 결과의 정수 점수 해시를 다음 배포 검증의 기준으로 설정한다.

```bash
EXPECTED_HTTP_PREDICTION_SHA256="$(python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"]) / "http_validation"
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
assert summary["count"] == 400 and summary["all_gates_passed"]
rows = [json.loads(line) for line in
        (root / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
assert len(rows) == 400
canonical = "".join(
    f"{r['essay_id']}\t{r['official_scores']['content']}\t"
    f"{r['official_scores']['organization']}\t{r['official_scores']['expression']}\n"
    for r in rows
)
print(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
PY
)"
export EXPECTED_HTTP_PREDICTION_SHA256
```

### 7. Docker 제출

```bash
docker login
GPU=all bash docker_release.sh push
```

`push`는 이미지를 다시 빌드하고, 해당 이미지로 400편의 출력 형식과 정수 점수 해시를 검증한 뒤 업로드한다. 업로드 없이 같은 검증만 실행하려면 `push` 대신 `verify`를 사용한다.

완료되면 출력되는 `release.txt`의 **`submission_url=docker://계정/저장소:태그`** 값을 제출한다. 이 파일은 `main_code_submission/results/release_final/<태그>_<실행시각>/`에 저장되며 이미지 digest와 예측 해시도 포함한다.

다른 GPU 서버에서 업로드한 이미지를 검증할 때는 저장소와 공식 validation 원본, `release.txt`를 준비하고 실행한다.

```bash
DATA_FILE="/absolute/path/to/official_validation.jsonl" GPU=all \
  bash code_for_docker_check_otherserv/check_uploaded_final.sh \
  /absolute/path/to/release.txt
```
