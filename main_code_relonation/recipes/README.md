# 근거 모델 recipe

기술서 기준 실행은 [r18_report_qwen35.json](r18_report_qwen35.json)을 사용한다.
Qwen3.5-9B BF16, attention q/k/v/o LoRA r32/alpha64/dropout0.05,
batch 2 × 누적 16, 학습률 4e-5, 2 epoch, warmup 0.05 설정이다.
입력 길이는 8192, 생성 상한은 2048, 점수는 fixed다.

| 파일 | 용도 |
|---|---|
| r18_report_qwen35.json | 기술서 기준 Qwen/v4 실행, 백본 revision 고정 |

`main_code_relonation.train`과 `infer`의 `--recipe` 기본값도 r18이다.

상대 rationale_prompt_file은 recipe 파일 위치 기준으로 해석한다.
학습된 어댑터의 prompt 원문·ID·SHA-256 sidecar는 추론에서도 동일하게 사용한다.
현재 v4와 기술서 prompt_8의 동일성은 미확인이다.

11,600편의 2 epoch를 현재 Python Trainer로 실행하면 726 step / warmup 37이다.
기술서의 725 / 36과의 차이는 Python 수정 금지 조건에 따라 남겨 두었다.
