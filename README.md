# 2026 AI 말평 — 글쓰기 채점 능력 평가

## Method

Qwen3.5-9B 백본에 채점 LoRA 8개와 근거 생성 LoRA 1개를 결합한다. 채점 모델은 다음 네 요소를 함께 학습한다.

- **A:** 5등급 분포의 기댓값으로 영역 점수를 예측하고 MSE + CE로 학습한다.
- **B:** 영역별 Soft-Spearman 손실을 추가한다(가중치 0.2, 온도 0.5).
- **C:** 평가자 준거 라벨을 `rater_set` 보조 head로 학습한다(expected MSE 0.25 + rater CE 0.25).
- **D:** 구성 영역에 문단 평균 pooling을 적용한다.

시드 42~49의 최종 연속 예측을 등가중 평균한 뒤 **SMR(`average_matched`)**을 한 번 적용한다. 근거 모델은 Gemma4-26B-A4B-it AWQ teacher의 생성 데이터로 학습하며, 추론 시 확정 점수를 유지하면서 근거만 생성한다. 프롬프트는 [rationale_prompt_v4.txt](main_code_relonation/prompts/rationale_prompt_v4.txt)를 사용한다.

| 설정           | 채점                                                  | 근거 생성                                                                    |
| -------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------- |
| 레시피         | [report_abcd.json](main_code/configs/report_abcd.json) | [r18_report_qwen35.json](main_code_relonation/recipes/r18_report_qwen35.json) |
| 백본           | Qwen3.5-9B, BF16                                      | 동일 백본 공유                                                               |
| LoRA           | r32 / alpha64, attention + MLP                        | r32 / alpha64, q/k/v/o                                                       |
| 학습 데이터    | 11,600편                                              | 11,600편                                                                     |
| 학습량         | 시드별 1,104 steps                                    | 2 epochs                                                                     |
| 배치 × 누적   | 32 × 1                                               | 2 × 16                                                                      |
| 학습률         | LoRA 4e-5 / 출력층 2e-4                               | 4e-5                                                                         |
| 최대 입력 길이 | 4,096                                                 | 8,192                                                                        |

## Installation

Python 3.13, CUDA 13.0 드라이버, GPU 메모리 80GB 이상이 필요하다(채점 학습 peak 약 70GiB, RTX PRO 6000 96GB에서 확인).

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-train.txt -r requirements-eval.txt \
  -r main_code_relonation/requirements.txt \
  transformers==5.14.0 peft==0.19.1 accelerate==1.14.0
pip install pytest  # 테스트 실행 시
```

teacher 근거 생성용 vLLM은 별도 환경에 설치한다.

```bash
python -m venv .venv-vllm
.venv-vllm/bin/pip install vllm==0.25.1 transformers==5.14.0 ninja
```

Docker 제출에는 Docker Engine과 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)이 필요하다.
Qwen3.5-9B는 학습 시 Hugging Face 기본 cache(`~/.cache/huggingface/hub`)에 내려받으며, 이미지 빌드도 이 위치에서 가중치를 복사한다.

## Data

두 종류의 자료가 필요하다. 모두 국립국어원 사용 신청과 승인이 필요하다.

- **대회 자료:** [글쓰기 채점 능력 평가 말뭉치 2026](https://kli.korean.go.kr/benchmark/taskOrdtm/taskDownload.do?taskOrdtmId=205&clCd=ING_TASK&subMenuId=sub02)의 train·validation JSONL
- **원천 자료:** 국립국어원 모두의 말뭉치의 「글쓰기 채점 자료 말뭉치 2024」, 「글쓰기 채점 자료 말뭉치 2023(2)」, 「글쓰기 채점 자료 말뭉치 2023(1)」

압축을 풀어 다음 위치에 둔다.

```text
main_code/datasets/raw_dataset/
├── official_competition/
│   ├── 글쓰기채점능력평가2026_train.jsonl
│   └── 글쓰기채점능력평가2026_validation.jsonl
├── NIKL_GRADING WRITING DATA 2023/       # GWGR*.json
├── NIKL_GRADING WRITING DATA 2023(2)/    # GWGR*.json
└── NIKL_GRADING WRITING DATA 2024/       # GWGR*.json
```

- `official_competition/`에는 `*train.jsonl`과 `*validation.jsonl`이 각각 하나만 있어야 한다.
- 원천 자료 폴더는 이름이 `NIKL_GRADING WRITING DATA`로 시작하는 폴더가 정확히 3개여야 하고, 각 폴더 바로 아래에 `*.json`이 있어야 한다. 배포 폴더명을 그대로 쓰면 된다.

```bash
bash scripts/prepare_data.sh
```

학습 11,600편과 검증 400편이 `main_code/datasets/processed_dataset/`에 생성된다.

<details>
<summary>근거 학습 데이터 생성</summary>

이미 생성한 데이터가 있으면 생략한다. teacher 모델과 고정 revision을 지정한다. 아래는 e5 근거 데이터 생성에 사용한 teacher다. teacher 서버는 GPU 여유 메모리 80GiB를 요구하며 `TEACHER_MIN_FREE_GIB`, `TEACHER_GPU_UTIL`로 조정한다.

```bash
# 터미널 1: teacher 서버
VLLM_PYTHON=.venv-vllm/bin/python \
  bash scripts/prepare_data.sh teacher google/gemma-4-26B-A4B-it 4d7ae4984b7db7de8f8457170b3f1a419ee76d52

# 터미널 2: 근거 생성
bash scripts/prepare_data.sh generate google/gemma-4-26B-A4B-it 4d7ae4984b7db7de8f8457170b3f1a419ee76d52
```

결과는 `main_code_relonation/results/report_rationale_v4/pseudo_train.jsonl`에 저장된다. 사람 점수에 SMR을 적용한 정수 점수를 조건으로 생성하고, 파싱·점수 복사·QC에 실패한 근거는 학습에서 제외한다. 별도 LLM judge 필터는 쓰지 않는다. 같은 명령을 다시 실행하면 실패한 항목만 재생성한다. 생성 완료 후 teacher 서버를 종료하고 학습한다.

</details>

## Training

```bash
# 채점 모델: 시드 42~49 순차 학습
GPU=0 bash scripts/train.sh score

# 근거 생성 모델
GPU=0 bash scripts/train.sh rationale
```

채점 모델은 시드당 약 5.5시간(RTX PRO 6000 96GB)이 걸리며 끝난 시드는 다시 실행할 때 건너뛴다. 별도 근거 데이터를 사용하려면 `bash scripts/train.sh rationale /path/to/pseudo.jsonl`로 지정한다. 결과는 `results/report_abcd/`에 저장된다. `OUT_ROOT`로 출력 경로를 변경할 수 있다.

## Docker Submission

```bash
# 이미지 빌드 및 공식 validation 400편 검증
bash scripts/submit.sh build report-abcd-v1

# Docker Hub 업로드
docker login
bash scripts/submit.sh push USER/writing-scorer report-abcd-v1
```

`USER/writing-scorer`를 공개 Docker Hub 저장소로 바꾼다. 스크립트가 매니페스트와 예측 해시를 관리하고, 업로드 전 동일한 400편 결과를 재검증한다. 공식 validation 원본은 `official_competition/`에서 찾으며, 다른 위치에 있으면 `DATA_FILE`로 지정한다.

완료 후 출력되는 `docker://USER/writing-scorer:report-abcd-v1`을 제출한다. 제출 정보는 `main_code_submission/results/release_final/`의 `release.txt`에 저장된다.
