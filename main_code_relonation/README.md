# 근거 생성 모델 (`main_code_relonation`)

> **[2026-08-25 정정]** 이 문서가 "확정"이라 적은 채점 모델(c02 / `skt/A.X-4.0-Light`)은
> 최종 제출본이 **아니다**. 최종 제출본은 `Qwen/Qwen3.5-9B` 8 seed 앙상블이며 근거 모델도
> Qwen student(v4 prompt)다. 아래 내용은 그 결정에 이르는 과정의 기록으로 읽을 것.
> 현재 확정 구성은 저장소 루트 ``README.md``와 `python show_submission_settings.py`가 기준이다.

폴더명 `relonation`은 기존 artifact 경로 호환 때문에 유지한다. 이 코드는 점수를 새로 예측하지
않는다. 이미 확정된 C/O/E 점수를 조건으로 제출 형식의 전체 JSON을 생성하는 A.X-4.0-Light
LoRA를 학습한다.

## 현재 확정 결과

현행 recipe의 전체 A/B 재현 사용자 진입점은 루트 `workflow.sh rationale-final`이다. 이 command가
내부 [`run_rationale_full11600_r32.sh`](run_rationale_full11600_r32.sh)을 호출한다. 기본 결과
leaf는 이미 존재하므로 새 `AB_ROOT`를 지정해야 한다.

```bash
cd /home/nlplab/research/gyop/2026_MalPyeong_tf
AB_ROOT=main_code_relonation/results/reproductions/rationale_full11600_$(date +%Y%m%d_%H%M%S) \
  bash workflow.sh rationale-final
```

이 실행은 다음 두 독립 arm을 비교한다.

| arm | teacher | student 결과 leaf |
|---|---|---|
| Model 1 | `google/gemma-4-26B-A4B-it` | `students/model1_gemma_teacher/<config>_gemma_full11600` |
| Model 2 | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | `students/model2_qwen_teacher/<config>_qwen_full11600` |

두 student 폴더는 상위 실험 폴더만 공유하고 leaf가 다르다. 이미 존재하는 student 폴더는
덮어쓰지 않고 즉시 중단한다. Teacher pseudo 파일은 arm별 폴더에 따로 저장하며, 같은 prompt와
request hash로 다시 실행하면 이미 성공한 ID를 건너뛰고 실패/미처리 ID만 재시도한다.

기본 결과 root는 다음과 같다.

```text
results/rationale_prompt_v1_teacher_ab_full11600_r32/
├── rationale_v1_gemma_teacher/pseudo_train.jsonl
├── rationale_v1_qwen_teacher/pseudo_train.jsonl
├── students/model1_gemma_teacher/<config>_gemma_full11600/
├── students/model2_qwen_teacher/<config>_qwen_full11600/
├── final_validation_submitted_scores.jsonl
└── students/<각 arm>/.../student_proxy_judge.jsonl
```

이번 실행은 예약 순서를 조정해 arm별 stage를 따로 완료했으므로 full runner가 마지막에 쓰는
`student_comparison.json`, `FULL_RUN_COMPLETE.json`과 중간 `dataset_ready.json`은 없다. 대신 실제
pseudo/train/validation/Judge artifact와 Docker release preflight를 대조해 Gemma arm을 선택했다.
최종 판정은
``FINAL_RATIONALE_SELECTION_20260812.md``에
고정한다. 없는 marker를 완료 증거로 간주하지 않는다.

## 6단계 실행 흐름

1. 두 teacher가 각각 train 11,600편 전체를 시도한다.
2. 생성 중 빠른 local parse/score-copy/grounding QC를 적용한다.
3. 각 teacher의 strict-pass 행만으로 각자 A.X student를 학습한다.
4. c02 score 모델의 validation 400편 제출 정수 점수 파일을 만든다.
5. 두 student가 같은 400편·같은 고정 점수로 전체 JSON을 생성하고 빠른 형식 검사를 통과한다.
6. 고정 Qwen local proxy Judge가 validation 400편만 평가하고 paired 결과를 비교한다.

학습 전에 느린 proxy Judge로 teacher 데이터를 거르지 않으며, 두 teacher의 accepted-ID 교집합도
만들지 않는다. 그러므로 두 arm은 teacher 품질뿐 아니라 teacher별 strict-pass 데이터의 양과
분포까지 포함한 end-to-end distillation 비교다. 역사 2,000편 pilot의 `judge-data`, `align`,
baseline 비교 subcommand는 재현용으로 controller에 남아 있지만 현행 full runner는 호출하지 않는다.

## Teacher 데이터 계약

원천 train universe는 public validation 400편을 제외한 NIKL 11,600편이다.

- `official_train`: 2,000편
- `origin_pool_extra`: 9,600편
- validation/external: 0편

각 행의 인간 소수 C/O/E를 ``average_matched``로
정수화해 teacher prompt의 `[고정 predicted_score]`와 JSON skeleton에 넣는다. 이는 Docker에서
score 모델의 연속 출력이 최종 정수로 바뀐 뒤 rationale 모델에 들어가는 표면과 맞추기 위한
조건화이며, score 모델 자체의 train label을 정수화한다는 뜻은 아니다.

Teacher는 rationale 문자열만 따로 내는 것이 아니라 아래 형태의 JSON 객체 전체를 직접 생성한다.

```json
{"content":{"score":4,"rationale":"..."},"organization":{"score":3,"rationale":"..."},"expression":{"score":3,"rationale":"..."}}
```

기본 3회 생성 뒤에도 실패하면 동일한 엄격 QC를 유지한 full-JSON repair를 최대 2회 수행한다.
다섯 번 모두 실패한 행은 audit에 남기고 다음 행으로 진행한다. 학습 파일에는 다음을 모두 만족한
행만 들어간다.

- JSON parse 성공과 C/O/E 전체 존재
- skeleton의 고정 score를 정확히 복사
- 세 rationale가 비어 있지 않고 서로 동일하지 않음
- 각 rationale 180자 이하
- 직접 인용이 실제 essay의 연속 부분 문자열
- 보이지 않는 문단/줄바꿈을 조직 근거로 사용하지 않음

전체 pass-rate를 임의 하한으로 맞추려고 QC를 완화하지 않는다. 이번 artifact의 strict-pass는
Gemma 11,600/11,600행, Qwen 9,230/11,600행이며 그 행들만 각각 학습했다. Full runner를 새
`AB_ROOT`에서 실행하면 `dataset_ready.json`도 쓰지만, 현재 선택의 권위는 실제 pseudo manifest와
최종 선택 문서다. API/server/CUDA 오류로 vLLM이 종료되면 행 오류로 숨기지 않고 stage 전체를
중단한다.

## Student 학습 계약

현행 recipe는 [`r12_ax_lora_fixed_prompt_v1.json`](recipes/r12_ax_lora_fixed_prompt_v1.json)이다.

- base: `skt/A.X-4.0-Light` pinned revision
- BF16 base, 4-bit/QLoRA 사용 안 함
- LoRA r32/alpha64/dropout 0.05
- target: Q/K/V/O projection
- batch 1, gradient accumulation 32, 2 epochs
- greedy generation, max input 8,192, max new tokens 512
- score mode `fixed`, prompt v1

Teacher별 strict-pass 행 수를 `N`이라 하면 optimizer step은
`2 × ceil(N / 32)`다. 두 teacher의 통과 행 수가 다르면 step 수도 달라지며, 결과 해석 때 데이터
수와 함께 기록한다. 실제 Gemma student는 726 step, Qwen student는 578 step이다. c02 score
LoRA는 attention+MLP target이지만 rationale LoRA는 Q/K/V/O만 사용한다. 제출 shared loader가
요구하는 score target의 부분집합이므로 Docker 코드를 바꿀 필요가 없다.

학습된 `final_adapter/`에는 다음 prompt/runtime sidecar가 필수다.

- `rationale_prompt.txt`
- `rationale_runtime_config.json`
- `adapter_config.json`, `adapter_model.safetensors`, tokenizer closure

## Validation과 승자 선택

두 student는 c02의 `average_matched` 제출 정수 점수 400개를 똑같이 조건으로 받았다. 400/400
parse, score-copy 불변과 ID 순서는 hard gate이고, nonempty·길이·인용 grounding은 별도 진단
집계다. 그 뒤 `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` local proxy Judge를 같은 revision·prompt로
실행했다.

Proxy Judge는 공식 Qwen3.6 계열 실행을 근사하지만 공식 Q4_K_M binary와 같지 않다. Gemma는
mean `4.6779166667`, Qwen은 `4.5927083333`이었고 400편 paired win/tie/loss는
`189/88/123`이라 Gemma를 선택했다. 이 값은 standalone student 출력의 local proxy 결과이며
pushed Docker rationale나 공식 Judge의 exact 점수가 아니다. Submission engine에서는 별도로
400편 형식과 점수 hash를 검증했다.

## Prompt와 Docker 결합

- [`prompts/baseline_prompt.txt`](prompts/baseline_prompt.txt): legacy `rationale_ax_v2` 전용
- [`prompts/rationale_prompt_v1.txt`](prompts/rationale_prompt_v1.txt): 현행 full 실험과 배포 모델
- ``RATIONALE_PROMPT_V1_METHOD.md``: Judge/논문 근거와 제약

Prompt는 모델 artifact다. `<<FIXED_SCORE_LINES>>`, `<<OUTPUT_SKELETON>>`,
`<<PROMPT_TEXT>>`, `<<ESSAY_TEXT>>`가 각각 정확히 한 번 있어야 한다. 학습 뒤에는 adapter에
저장된 prompt 원문과 SHA-256이 authoritative하며, research inference와 Docker는 mismatch를
거부한다.

이 폴더의 `Dockerfile`/`serve.py`는 score backend가 없는 연구용 joint generator다. 최종 제출은
사용하지 않는다. [`main_code_submission`](../main_code_submission/) 엔진이 c02 점수를 먼저
`average_matched` 정수로 확정한 뒤, 같은 A.X CausalLM에서 score adapter와 선택 rationale
adapter를 순차 전환한다. 생성 모델이 JSON에 다른 score를 내더라도 최종 응답 score는 c02의
확정 정수를 다시 복사한다.

## 개발용 단일 pipeline

개별 모듈 smoke나 과거 artifact 재현에는 `workflow.sh rationale`을 사용할 수 있다.

```bash
GPU=0 STAGE=teacher bash workflow.sh rationale
GPU=0 STAGE=generate bash workflow.sh rationale
GPU=0 STAGE=train bash workflow.sh rationale
GPU=0 STAGE=verify bash workflow.sh rationale
```

이 경로는 확정 Gemma/Qwen full 비교의 재현 경로를 대체하지 않는다. 최종 결과와 provenance는
``results/README.md``, prompt 파일 규칙은
``prompts/README.md``, recipe 구분은 ``recipes/README.md``를
따른다.
