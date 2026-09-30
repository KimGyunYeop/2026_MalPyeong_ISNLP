# 채점 모델 (`main_code`)

`content` / `organization` / `expression` 세 영역의 점수를 학습하고 추론한다.
연속 원점수와 제출용 정수 표면을 함께 저장하며, 공개 validation 400편의 공식
RMSE / Spearman을 계산한다.

전체 재현 절차와 최종 제출 구성은 저장소 루트의 `README.md`가 기준이다.
현재 실행 설정은 `configs/report_abcd.json`이다.

## 실행

```bash
# 학습 — 한 시드 예시. 전체 8시드 실행은 루트 README 참고
CUDA_VISIBLE_DEVICES=0 python -m main_code.train \
  --config main_code/configs/report_abcd.json \
  --dataset-root /path/to/main_code/datasets \
  --seed 42 --output-dir results/report_abcd/score_s42

# 추론 — 체크포인트 하나로 400편 평가
python -m main_code.infer --checkpoint <checkpoint> --output-dir <out>

# 데이터 준비 (공식·원천 자료를 main_code/datasets/raw_dataset/에 배치)
bash main_code/build_datasets.sh
```

## 구성 요소

| 파일 | 역할 |
|---|---|
| `config.py` | `RegressionConfig` dataclass. 기술서 실행 시 `report_abcd.json`을 명시한다 |
| `datasets.py` | 입력 조립, 에세이 표면, 문단·문장 span, collator |
| `models.py` | 백본 + LoRA + 점수 head, 손실 전부 |
| `train.py` / `infer.py` | 학습·추론 진입점 |
| `prepare_data.py` | 원천 자료에서 학습·검증 JSONL 생성 |
| `postprocess.py` | 제출 점수 변환(`average_matched` 등) |
| `official_metrics.py` | 공지 원문 그대로의 RMSE / Spearman. **수정 금지** |
| `paragraph_boundary.py` | 문단 경계 후보·라벨 (요소 D와 문단 보조 과제가 쓴다) |
| `quantization_objectives.py` | 정수 출력 제약을 학습 손실로 다루는 목적함수 |

## 설정 preset

| 파일 | 내용 |
|---|---|
| `configs/report_abcd.json` | **기술서 기준 실행 설정, Qwen revision 고정** |
| `configs/baseline.json` | 기술서 기준 채점 모델(baseline). report_abcd에서 A·B·C·D만 끈 설정 |

`configs/README.md`에 각 preset의 필드 규약이 있다.

## 테스트

```bash
python -m pytest main_code/tests/
```
