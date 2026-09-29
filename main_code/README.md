# 채점 모델 (`main_code`)

`content` / `organization` / `expression` 세 영역의 점수를 학습하고 추론한다.
연속 원점수와 제출용 정수 표면을 함께 저장하며, 공개 validation 400편의 공식
RMSE / Spearman을 계산한다.

전체 재현 절차와 최종 제출 구성은 저장소 루트의 `README.md`가 기준이다.
현재 설정을 눈으로 확인하려면 루트에서 다음을 실행한다.

```bash
python show_submission_settings.py
```

## 실행

```bash
# 학습 — 인자 없이 돌리면 제출 구성(ABCD) 멤버 s42가 나온다
python -m main_code.train --output-dir results/bbq35_e5_s42

# 추론 — 체크포인트 하나로 400편 평가
python -m main_code.infer --checkpoint <checkpoint> --output-dir <out>

# 데이터 준비 (원본 JSONL이 datasets/ 에 있어야 한다)
bash main_code/build_datasets.sh
```

## 구성 요소

| 파일 | 역할 |
|---|---|
| `config.py` | `RegressionConfig` 단일 dataclass. 기본값이 곧 제출 설정이다 |
| `datasets.py` | 입력 조립, 에세이 표면, 문단·문장 span, collator |
| `models.py` | 백본 + LoRA + 점수 head, 손실 전부 |
| `train.py` / `infer.py` | 학습·추론 진입점 |
| `prepare_data.py` | 원천 자료에서 학습·검증 JSONL 생성 |
| `postprocess.py` | 제출 점수 변환(`average_matched` 등) |
| `official_metrics.py` | 공지 원문 그대로의 RMSE / Spearman. **수정 금지** |
| `paragraph_boundary.py` | 문단 경계 후보·라벨 (요소 D와 문단 보조 과제가 쓴다) |
| `quantization_objectives.py` | 정수 출력 제약을 학습 손실로 다루는 목적함수 |
| `configs/confirmed_final.json` | 최종 제출 설정. `RegressionConfig()` 기본값과 동일하다 |

## 설정 preset

| 파일 | 내용 |
|---|---|
| `configs/confirmed_final.json` | **최종 제출 구성** |
| `configs/confirmed_y1.json` | 초기 회귀 head 계보 (동결) |
| `configs/baseline.json` | 2026-08-10 이전 legacy baseline (동결) |

`configs/README.md`에 각 preset의 필드 규약이 있다.

## 테스트

```bash
python -m pytest main_code/tests/
```
