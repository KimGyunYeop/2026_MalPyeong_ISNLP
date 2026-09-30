# 근거 prompt 파일

현재 Qwen 레시피는 [rationale_prompt_v4.txt](rationale_prompt_v4.txt)를 사용한다.
기술서 prompt_8과 같은 원문인지는 미확인이므로 이름이나 해시를 바꾸지 않는다.
baseline_prompt.txt는 prompt를 선언하지 않은 recipe의 기본값이고,
rationale_prompt_v1.txt는 prompts.py의 prompt ID 판정에 쓰이므로 함께 둔다.

각 template에는 다음 sentinel이 정확히 한 번씩 있어야 한다.

- `<<FIXED_SCORE_LINES>>`
- `<<OUTPUT_SKELETON>>`
- `<<PROMPT_TEXT>>`
- `<<ESSAY_TEXT>>`

load_prompt_template()은 파일 끝 개행만 제거한다. v4의 runtime SHA-256은
`67df14d2264531e88efde762c7e323b7b2f9d537b57fc07ad7e636d7faf945d1`이다.
파일 자체의 해시와 runtime 해시는 끝 개행 때문에 다를 수 있다.

학습 시 prompt 원문은 final_adapter/rationale_prompt.txt와
rationale_runtime_config.json에 저장된다. 제출 매니페스트와 Docker는 해당 어댑터의
원문·ID·SHA-256을 그대로 사용한다. 기존 학습 artifact의 prompt를 교체하지 않는다.
