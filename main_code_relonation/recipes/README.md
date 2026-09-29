# 근거 모델 recipe

| 파일 | 상태 | 핵심 설정 |
|---|---|---|
| `r11_ax_lora_fixed.json` | legacy control | NF4 QLoRA, r16/a32, baseline prompt |
| `r12_ax_lora_fixed_prompt_v1.json` | 현행 선택·배포 recipe | BF16 LoRA, r32/a64, prompt v1 |

r12는 A.X-4.0-Light의 Q/K/V/O projection에 LoRA를 학습한다. Score LoRA의
attention+MLP target보다 좁지만 부분집합이므로 제출 shared CausalLM loader와 호환된다. 두
teacher arm은 같은 r12 파일과 seed를 쓰고, 입력 `pseudo_train.jsonl`과 그에 따른 row/step 수만
달라질 수 있다.

Recipe의 상대 `rationale_prompt_file`은 recipe 파일 위치를 기준으로 resolve된다. 학습 후에는
recipe보다 adapter sidecar가 runtime의 authoritative prompt다. 과거 결과를 재현할 때 current
default에 의존하는 partial JSON을 만들지 말고 당시의 `resolved_config.json`을 사용한다.

현행 사용자 진입점 `workflow.sh rationale-final`이 호출하는 full runner의 기본 recipe는 r12다.
다만 Python에서 bare
`RationaleConfig(model_id=...)`를 직접 만들면 legacy 호환용 baseline prompt와 r16/a32가 남아
있으므로, 최종 구조를 import해 쓰는 코드는 r12를 `load_config()`로 읽거나 선택 adapter의 runtime
sidecar를 bind해야 한다.
