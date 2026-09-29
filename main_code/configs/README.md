# 점수 모델 설정

> **[2026-08-25 정정]** 이 문서가 "확정"이라 적은 채점 모델(c02 / `skt/A.X-4.0-Light`)은
> 최종 제출본이 **아니다**. 최종 제출본은 `Qwen/Qwen3.5-9B` 8 seed 앙상블이며 근거 모델도
> Qwen student(v4 prompt)다. 아래 내용은 그 결정에 이르는 과정의 기록으로 읽을 것.
> 현재 확정 구성은 저장소 루트 ``README.md``와 `python show_submission_settings.py`가 기준이다.

이 폴더는 점수 모델의 재현 가능한 입력 설정만 보관한다. 연구 학습의 기본값과 실제 Docker에
배포한 checkpoint는 서로 다른 개념이다.

## 설정 구분

- [`confirmed_final.json`](confirmed_final.json): 현재 **score 모델** 연구용 기본 설정이다.
  `RegressionConfig()`와 config 없는 `train.py`가 이 설정과 일치한다. 최종 score 승자
  `c02_avg_mse025/a228dbe60ed8`의 `resolved_config.json`과 byte 단위로 같으며 SHA-256은
  `fa3e280b368d0dc8ce79c977574cf275ab03053f95a12c0648ba4870fd224448`이다.
- [`confirmed_y1.json`](confirmed_y1.json): 이전 expanded-Y1 재현용 완전 고정 설정이다.
  `--config main_code/configs/confirmed_y1.json`을 명시할 때만 사용한다.
- [`baseline.json`](baseline.json): 과거 baseline 재현용 고정 설정이다. `--baseline` 또는
  `--config main_code/configs/baseline.json`을 명시할 때만 사용한다.
- 최종 제출 모델: 이 폴더에서 자동으로 정하지 않는다. 현행 선택은 루트
  [`docker_release.sh`](../../docker_release.sh)가 checkpoint와 rationale artifact를 명시적으로
  검사하고 source manifest를 결정론적으로 만든다.

현재 final은 대회 train 11,600편만 사용한 seed 43의 BF16 일반 LoRA(r32/alpha64) 설정이다.
direct 5-class score head에 trait-average MSE 0.25를 더하고, `official_matched_rmse`로 고른
step 736 checkpoint를 사용한다. QLoRA와 외부 데이터는 쓰지 않는다. 확정 checkpoint는
[`best_checkpoint_official_matched_rmse`](../results/new_proposed/y6_final_combo_noext_s43_v1/c02_avg_mse025/ax4_light/a228dbe60ed8/best_checkpoint_official_matched_rmse)다.

공식 raw validation 400편의 batch-1 제출 표면은 RMSE `0.416823637099`, Spearman
`0.759736661325`이며 batch-16과 제출 정수 1,200개가 같다. 최종 pushed Docker도 같은 정수열을
재현했고 HTTP parity SHA-256은 `ca9b4244e95b0d1694d1aa85bd68194c0f1140a23fa2eee34f8f33c2e94047a7`이다.
역사 Y6 manifest는 이 config의 배포 기록이 아니다.

`confirmed_final.json`의 `model_revision`은 원래 c02 run과 같은 `main`이다. 따라서 recipe 구조와
checkpoint 설정은 재현하지만, 미래에 remote `main`이 이동한 뒤 base weight까지 byte-exact하게
재학습하려면 현재 Docker가 고정한 A.X snapshot
`ba21c20ea1b31ded1ec3e2fb432335077dc4be98`을 별도로 명시해야 한다.

## 해석 규칙

부분 JSON은 독립 recipe가 아니다. 빠진 필드는 현재 `RegressionConfig()` 기본값으로 채워지므로,
과거 실험을 재현할 때는 당시의 완전한 `resolved_config.json` 또는 모든 필드를 가진 고정 config를
사용한다. 정확한 option과 validation 제약은
``OPTIONS_REFERENCE.md``를 따른다.

재현 경로는 다음과 같다.

```text
prepared dataset
→ full resolved config
→ train.py
→ results/<suite>/<group>/<case>/<model>/<config_id>/
→ selected checkpoint
→ infer.py
→ docker_release.sh의 검증된 source manifest
```

새로운 확정 설정을 추가할 때는 model ID와 revision, 데이터 fingerprint, seed, 학습 step,
checkpoint 선택 지표, 후처리 surface를 생략하지 않는다. 점수 수식과 지표 정의는
``METHOD_AND_METRIC_CONTRACT.md``를 따른다.
