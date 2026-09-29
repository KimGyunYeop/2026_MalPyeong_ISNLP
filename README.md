# 2026 AI 말평 — 글쓰기 채점 능력 평가

모델 기술서의 **Qwen3.5-9B · ABCD · 8시드 등가중 평균 · SMR** 설정을 사용한다.
학습은 기존 `main_code.train`과 `main_code_relonation.train`을 사용하고,
제출 매니페스트는 기존 [docker_release.sh](docker_release.sh)로 생성한다.
성능 평가는 별도로 진행한다.

## 기술서 기준 설정

| 항목           | 채점                                                  | 근거                                                                         |
| -------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------- |
| 레시피         | [report_abcd.json](main_code/configs/report_abcd.json) | [r18_report_qwen35.json](main_code_relonation/recipes/r18_report_qwen35.json) |
| 백본           | Qwen3.5-9B, BF16                                      | 같은 백본 공유                                                               |
| LoRA           | r32 / alpha64 / dropout0.05, attention+MLP            | r32 / alpha64 / dropout0.05, q/k/v/o                                         |
| 학습량         | 11,600편, 1,104 step                                  | 11,600편, 2 epoch                                                            |
| 배치 × 누적   | 32 × 1                                               | 2 × 16                                                                      |
| 학습률         | LoRA 4e-5, 출력층 2e-4                                | 4e-5                                                                         |
| 최대 입력 길이 | 4,096                                                 | 8,192                                                                        |
| warmup 비율    | 0.05                                                  | 0.05                                                                         |
| 시드           | 42~49                                                 | 42                                                                           |

- A: 5등급 분포 기댓값, MSE + CE.
- B: 영역별 Soft-Spearman, 가중치 0.2, 온도 0.5.
- C: rater_set, expected MSE 0.25 + rater CE 0.25.
- D: 구성 영역 paragraph_mean.
- 8개 연속 예측을 각각 0.125로 평균한 후 SMR을 한 번 적용한다. 점수 offset은 0이다.
- 근거는 확정 점수를 유지하며 최대 2,048 tokens, 첫 생성은 temperature 0 / top_p 1이다.

두 신규 레시피는 확인에 사용한 Qwen snapshot
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`를 고정한다.
기존 confirmed_final.json도 ABCD이며 revision은 main이다.
과거 AB/A.X 체크포인트·해시·성능은 새 실행의 기본값으로 사용하지 않는다.

## 환경 및 데이터

```bash
pip install -r requirements-train.txt -r requirements-eval.txt
```

prepared dataset은 `DATASET_ROOT/processed_dataset/{train,validation}.jsonl`에 둔다.
train 11,600편에는 C 보조 손실에 필요한 원천 준거 라벨이 있어야 하고 validation은 400편이다.
main_code/build_datasets.sh로 새로 만들 때는 해당 스크립트가 요구하는 공식·원천 자료를
main_code/datasets/raw_dataset/에 준비한다.

확인에 사용한 환경은 Python 3.13.9, torch 2.13.0+cu130, transformers 5.14.0,
peft 0.19.1이다. 로컬 모델 cache만 사용하려면
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`을 지정한다.

## 학습 명령

아래 명령은 사용자가 별도로 실행할 때 전체 학습을 시작한다.

```bash
for seed in 42 43 44 45 46 47 48 49; do
  CUDA_VISIBLE_DEVICES=0 python -m main_code.train \
    --config main_code/configs/report_abcd.json \
    --dataset-root /path/to/main_code/datasets \
    --seed "$seed" --output-dir "results/report_abcd/score_s${seed}" || break
done

CUDA_VISIBLE_DEVICES=0 python -m main_code_relonation.train \
  --recipe main_code_relonation/recipes/r18_report_qwen35.json \
  --train-file /path/to/accepted_teacher_11600.jsonl \
  --output-dir results/report_abcd/rationale
```

채점은 기술서의 64 step 평가 주기 설정을 유지하고 별도 gold overlay를 적용하지 않는다.
근거 학습 입력은 prompt·고정 점수 검사를 통과하는 accepted 11,600편을 준비한다.
위 명령의 출력은 `results/report_abcd/score_s42`~`score_s49`,
`results/report_abcd/rationale`이다. 이미 사용한 출력 디렉터리는 재사용하지 않는다.
경로를 바꾸면 아래 배포 명령의 OUT_ROOT 또는 개별 어댑터 경로도 함께 지정한다.

teacher 생성에는 기술서의 Gemma4-26B-A4B-it AWQ 경로·revision과 prompt_8 원문을
main_code_relonation/run_rationale_pipeline.sh에 명시해야 한다.
현재 보존된 v4 prompt와 prompt_8의 동일성은 미확인이므로 임의로 이름을 바꾸지 않았다.

## 제출 매니페스트

학습 완료 후 다음 명령으로 실제 가중치 해시·prompt sidecar를 담은 매니페스트를 만든다.
이 명령은 Docker 실행이나 성능 평가를 하지 않는다.

```bash
bash docker_release.sh check
```

경로는 SCORE_CHECKPOINTS, RATIONALE_ADAPTER, OUT_ROOT, PYTHON_BIN으로 지정한다.
완료된 학습 산출물이 없으면 중단한다. 실제 빌드·평가·배포는 별도 작업이다.
생성된 매니페스트는 `main_code_submission/results/report_manifests/`에 저장하며,
가중치·실험 결과와 함께 Git에서 제외한다.
