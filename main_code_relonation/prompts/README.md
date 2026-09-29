# 근거 prompt 파일

이 폴더의 `.txt`는 설명 문서가 아니라 학습·추론 artifact다.

| 파일 | 용도 |
|---|---|
| `baseline_prompt.txt` | sidecar가 없는 legacy `rationale_ax_v2` 재현 |
| `rationale_prompt_v1.txt` | 현행 11,600편 Gemma/Qwen distillation과 선택된 Docker 모델 |

각 template에는 다음 sentinel이 정확히 한 번씩 있어야 한다.

- `<<FIXED_SCORE_LINES>>`
- `<<OUTPUT_SKELETON>>`
- `<<PROMPT_TEXT>>`
- `<<ESSAY_TEXT>>`

`prompts.py`는 정규식의 lambda replacement로 네 슬롯을 한 번만 치환한다. Prompt나 essay 안의
문자열을 다시 template로 해석하지 않는다. 현재 공개 train/validation에는 sentinel과 section
marker 충돌이 없다.

`load_prompt_template()`은 저장소 텍스트 파일 끝의 개행만 `rstrip("\n")`한 문자열을 실제 model
prompt로 사용한다. 따라서 현행 runtime prompt SHA-256은 adapter sidecar와 같은
`9b343bf059c923e51f672bc7b4a6994d8092ae59f4a5fd9a881a66f170cb1c78`이며, 끝 개행을 포함한 source
file 자체에 `sha256sum`을 적용한 값과는 다르다.

새 prompt 버전을 만들 때 기존 파일을 수정하지 않는다. 새 `.txt`와 새 prompt ID를 만들고,
recipe에서 명시한 뒤 runtime template SHA-256 회귀 테스트를 추가한다. 학습이 끝나면 위 방식으로
읽은 정확한 runtime 원문을
`final_adapter/rationale_prompt.txt`와 `rationale_runtime_config.json`에 저장한다. 이후 research
inference, submission manifest와 Docker는 그 원문·ID·SHA가 모두 같아야 하며 mismatch는 오류다.

설계 근거와 Judge 계약은 ``RATIONALE_PROMPT_V1_METHOD.md``를
따른다.
