# 점수 모델 설정

기술서 재현에는 [report_abcd.json](report_abcd.json)을 사용한다.
Qwen3.5-9B snapshot을 고정한 ABCD 전체 설정이다.
기존 `python -m main_code.train --config main_code/configs/report_abcd.json`에
seed 42~49를 각각 지정한다. 전체 명령은 루트 [README](../../README.md)에 있다.

| 파일 | 용도 |
|---|---|
| report_abcd.json | 기술서 기준 실행용 ABCD, 고정 Qwen revision |
| confirmed_final.json | 현재 Python 기본값과 동일한 ABCD, revision main |
| confirmed_y1.json | 과거 Y1 실험 재현 |
| baseline.json | 과거 baseline 실험 재현 |

ABCD는 distribution 기댓값·MSE+CE, 영역별 Soft-Spearman 0.2,
rater_set 보조 손실 0.25+0.25, 구성 영역 paragraph_mean이다.
batch 32 × 누적 1, 최대 1104 step, 평가 주기 64 step, LoRA r32/alpha64를 사용한다.
제출에는 각 시드의 마지막 checkpoint를 연결하고 연속 예측 평균 후 SMR을 적용한다.

과거 A.X/c02 성능이나 가중치 해시는 현재 Qwen 설정에 속하지 않는다.
실제 제출 매니페스트는 학습 산출물에서 docker_release.sh check로 만든다.

report_abcd.json에는 현재 학습 설정 로더가 지원하는 필드만 둔다.
warmup 비율 0.05의 현재 계산값은 56 step이며 기술서 표기는 55다.
손실 수식 등 남은 구현 차이는 루트 README에 명시한다.
