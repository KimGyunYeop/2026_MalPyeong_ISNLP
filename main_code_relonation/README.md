# 근거 생성 모델

Qwen3.5-9B에 attention 전용 LoRA를 학습한다. 확정된 content/organization/expression
점수와 문제·학생 글을 입력받아 점수를 유지한 근거 JSON을 생성한다.
폴더명 relonation은 기존 artifact 경로 호환 때문에 유지한다.

현재 레시피는 [r18_report_qwen35.json](recipes/r18_report_qwen35.json)이다.
학습 데이터는 기술서의 Gemma4-26B-A4B-it AWQ teacher로 만든 accepted 11,600편을 사용한다.

저장소 루트에서 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m main_code_relonation.train \
  --recipe main_code_relonation/recipes/r18_report_qwen35.json \
  --train-file /path/to/accepted_teacher_11600.jsonl \
  --output-dir results/report_abcd/rationale
```

출력은 results/report_abcd/rationale/final_adapter이며 제출 매니페스트에 같은 경로를 연결한다.
학습·추론은 어댑터에 저장된 prompt 원문·ID·해시와 chat template을 공유한다.

teacher 생성 도구 [run_rationale_pipeline.sh](run_rationale_pipeline.sh)는 stage를 명시해 사용한다.
teacher/generate에는 TEACHER_MODEL, TEACHER_MODEL_REVISION, RATIONALE_PROMPT_FILE을 지정한다.
기술서의 prompt_8 원문과 현재 v4의 동일성은 미확인이다.
judge/verify는 별도 평가 작업이므로 이번 설정 정리에서는 실행하지 않는다.

현재 Python의 epoch 계산에서는 11,600편·2 epoch가 726 step이다(기술서 725).
