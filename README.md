# 2026 국립국어원 AI 말평 — 글쓰기 채점 능력 평가

학생 글의 **내용·구성·표현** 세 영역 점수를 예측하고, 각 점수에 대한 근거를 생성한다.
하나의 `Qwen3.5-9B` 백본을 공유하는 **채점 LoRA 어댑터**와 **근거 생성 LoRA 어댑터**로
구성된다.

이 저장소에는 제출물을 처음부터 재현하는 코드와 스크립트만 들어 있다.
대회 데이터는 배포가 제한되어 포함하지 않는다.

| | |
|---|---|
| 이미지 | `docker://gyunyeop/writing-scorer:qwen35-8seed-qwenrationale-nooffset-r4-20260825` |
| 백본 | `Qwen/Qwen3.5-9B` (채점·근거 공유) |
| 채점 | ABCD 구성 · 8 seed 등가중 앙상블 |
| 출력 | 영역별 1~5 정수 + 영역별 근거 문장 |

---

## 채점 모델의 구성 — A · B · C · D

기준 모델은 마지막 층 토큰 평균에 영역별 선형 출력층과 MSE만 쓴다. 여기에 네 요소를 얹은
것이 최종 채점 모델이다.

| | 요소 | 무엇을 바꾸는가 | 설정 |
|---|---|---|---|
| **A** | 점수 분포 예측 | 영역마다 실수 하나 대신 **1~5점 확률분포**를 내고 기댓값 `Σ k·p_k`를 점수로 쓴다 | `score_head=distribution` |
| **B** | 영역별 순위 학습 | 배치 안에서 예측 순위가 정답 순위에 가까워지도록 **Soft-Spearman** 손실을 더한다 | `listwise_loss=soft_spearman`, w=0.2 |
| **C** | 평가자별 보조 학습 | 익명 **평가자 2인**에 대응하는 출력 경로 2개로 9개 세부 평가지표 분포를 예측한다 | `detail_head_mode=rater_set`, w=0.25 |
| **D** | 문단 구간 평균 | 본문에 남은 **이중 공백**을 문단 단서로 삼아 구간 평균의 평균을 구성 영역 표현에 더한다 | `organization_pooling=paragraph_mean` |

### 각 요소를 넣은 이유

**A.** 학습 라벨은 평가자 점수를 집계한 값이라 `3.7` 같은 소수점이 대부분이다. 정수로
반올림하면 글 간 미세한 차이가 사라진다. 정답을 인접 두 등급에 나누어(`3.7` → 3에 0.3,
4에 0.7) 분포로 학습하고, 읽을 때는 기댓값을 써서 연속 점수를 유지한다.

**B.** 공식 지표의 절반이 Spearman이다. 오차만 줄이는 손실은 순위를 직접 개선한다는
보장이 없으므로, 미분 가능한 근사 순위로 순위 상관을 직접 최적화한다.

**C.** 원천 자료에는 영역 점수뿐 아니라 평가자별 9개 세부 평가지표 점수가 있다. 평가자
순서가 글마다 같은 사람을 가리키지 않으므로 출력 경로를 평가자에 고정하지 않고, 글마다
직접·교차 대응 중 손실이 작은 쪽을 쓴다. 최종 출력은 세 영역 점수 그대로이고 보조
출력층은 추론에 쓰지 않는다.

**D.** 구성 영역은 문단 구조를 본다. 그런데 대회 배포 형식은 문단을 이어 붙여 개행이
없다. 남아 있는 이중 공백으로 구간을 나누고 구간마다 같은 비중을 준다. 반영 강도 γ는
**0에서 시작하는 학습 스칼라 하나**라, 단서가 없는 글에서는 잔차가 0이 되어 기준 구성과
정확히 같아진다.

### 제출 점수 변환과 앙상블

공식 지표는 세 영역 점수의 **평균 하나**만 본다. 영역마다 독립으로 반올림하면 그 평균의
양자화 간격이 1이지만, 세 정수의 **합**을 연속 예측의 합에 맞추면 1/3로 줄어든다
(`average_matched`).

앙상블은 서로 다른 난수 시드로 학습한 **8개 어댑터의 등가중 평균**이다. 부분집합 중
최선을 고르면 검증셋에 대한 선택 편향이 붙으므로 8개를 전부 쓴다. 결합은 **정수화 이전
연속 점수**에서 하고 정수화는 한 번만 한다.

### 근거 모델

근거 문장 정답이 제공되지 않으므로, teacher 모델이 만든 근거를 QC로 거른 뒤 Qwen
student에 증류했다. 추론에서는 채점 모델이 점수를 먼저 확정하고, 근거 모델은 그 점수를
**고정 조건**으로 받아 설명만 생성한다(`score_mode=fixed`). 근거 생성이 실패해도 점수는
어떤 경우에도 응답에 실린다.

---

## 재현 절차

### 0. 환경

```bash
python -m venv .venv-train && . .venv-train/bin/activate
pip install -r requirements-train.txt      # 학습·추론
pip install -r requirements-eval.txt       # 평가
```

대회 데이터 원본을 `datasets/`에 두고 전처리한다.

```bash
bash main_code/build_datasets.sh
```

### 1. 채점 모델 — ABCD, 8 seed

**인자 없이 실행하면 ABCD 구성이 그대로 나온다.** 기본값이 곧 제출 설정이다.

```bash
python -m main_code.train --output-dir results/bbq35_e5_s42
```

학습 전에 설정을 확인하려면:

```bash
python show_submission_settings.py          # 사람이 읽는 요약

python -c "
import dataclasses, json
from main_code.config import RegressionConfig
a = dataclasses.asdict(RegressionConfig())
b = json.load(open('main_code/configs/confirmed_final.json'))
print('차이:', {k for k in set(a) | set(b) if a.get(k) != b.get(k)} or '없음')
"
```

`차이: 없음`이 나와야 한다.

8개 멤버는 `--seed`만 다르다.

```bash
for s in 42 43 44 45 46 47 48 49; do
  python -m main_code.train --seed "$s" --output-dir "results/bbq35_e5_s${s}"
done
```

체크포인트는 **마지막 step(1104)** 을 쓴다. 검증 400편에서 best-of-N을 고르면 선택 편향이
붙기 때문이다. GPU 1장 기준 seed당 약 5.5시간.

### 2. 근거 모델

**인자 없이 실행하면 제출 레시피(v4 프롬프트)가 기본값이다.**

```bash
python -m main_code_relonation.train \
  --train-file <teacher 증류 학습 파일> --output-dir results/rationale_v4_qwen35_student
```

teacher 근거 생성부터 student 학습까지 한 번에 돌리려면:

```bash
bash main_code_relonation/run_rationale_pipeline.sh
```

### 3. 두 모델을 합쳐 이미지 빌드 · 검증 · 배포

```bash
bash docker_release.sh check    # 아티팩트·지문·매니페스트 preflight
bash docker_release.sh verify   # 같은 image ID로 400편 HTTP 검증 (약 90분)
bash docker_release.sh push     # 400편 재검증을 통과할 때만 registry push
```

채점 8개 어댑터와 근거 어댑터를 하나의 백본 위에 싣는다. 인자 없이 돌리면 제출본과 같은
매니페스트를 만들고 제출본 예측 해시로 스스로를 검증한다. **모든 게이트가 fail-closed다**
— 하나라도 어긋나면 push하지 않는다.

체크포인트 경로는 `SCORE_CHECKPOINTS`, 근거 어댑터는 `RATIONALE_ADAPTER`로 덮어쓴다.

### 4. 업로드본 재검증 (다른 서버에서)

```bash
cd code_for_docker_check_otherserv
bash check_uploaded_final.sh /path/to/release.txt
```

registry에서 digest로 pull해 400편을 다시 돌린다. 느린 GPU에서는 근거 생성이
`deadline_seconds`에 걸려 강등될 수 있고, 그 경우에도 점수는 온전히 실린다.
`HTTP_DEGRADED_RESPONSE_BUDGET=3`을 주면 통과한다.

---

## 구조

```
main_code/              채점 모델 학습·추론
  config.py             모든 실행 옵션의 단일 원천. 기본값 = 제출 설정(ABCD)
  configs/confirmed_final.json   제출 resolved config (기본값과 동일해야 함)
  models.py             백본 + LoRA + 점수 head + 손실
  datasets.py           입력 조립, 에세이 표면, 문단·문장 span
  official_metrics.py   공지 원문 그대로의 RMSE/Spearman. 수정 금지
  postprocess.py        average_matched 정수 변환
main_code_relonation/   근거 모델 학습·추론
  prompts/rationale_prompt_v4.txt                제출본 근거 프롬프트
  recipes/r17_qwen35_lora_fixed_prompt_v4.json   제출본 학습 레시피
main_code_submission/   서빙 (FastAPI + transformers), Dockerfile
  engine.py             백본 1벌 공유 → 8어댑터 채점 → 근거
  degrade.py            근거 실패 시에도 점수는 반드시 싣는 경로
docker_release.sh       빌드 → 400편 검증 → push
code_for_docker_check_otherserv/   업로드본 재검증 번들
```

## 테스트

```bash
python -m pytest main_code/tests main_code_relonation/tests main_code_submission/tests -q
```

공식 지표 재현, `average_matched` 정수화, 근거 프롬프트 바인딩, 점수 유실 금지, 제출
매니페스트 구성을 코드로 고정한다. 대회 데이터나 빌드 staging이 없는 새 클론에서는
해당 테스트가 skip된다.
