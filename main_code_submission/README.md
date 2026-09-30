# 제출 Docker

현재 구성은 **Qwen3.5-9B 공유 백본 + ABCD 채점 LoRA 8개 + Qwen 근거 LoRA**다.
각 시드의 연속 예측을 같은 비중으로 평균하고 SMR을 한 번 적용한다.
integer_total_offset은 0이다. 확정 점수는 근거 생성 중 변경하지 않는다.

완료된 가중치에서 실제 제출 매니페스트를 만들려면 루트에서 실행한다.

```bash
bash docker_release.sh check
```

기본 채점 경로는 results/report_abcd/score_s42~score_s49/checkpoint,
근거 경로는 results/report_abcd/rationale/final_adapter다.
SCORE_CHECKPOINTS, RATIONALE_ADAPTER, OUT_ROOT, PYTHON_BIN으로 변경할 수 있다.

check는 ABCD 설정·시드·학습량과 prompt를 확인하고 실제 artifact 해시를 기록한다.
모델을 GPU에 올리거나 HTTP 평가를 실행하지 않는다.
가중치가 준비되지 않았으면 중단한다. 과거 AB/A.X 성능이나 해시는 새 manifest에 복사하지 않는다.

빌드·평가·배포는 나중에 별도로 실행한다. build_image.sh는 source manifest를 읽고 asset을
image에 내장한다. release_final.sh 및 docker_release.sh의 verify/push는 실제 평가를 포함하므로
이번 설정 작업에서 실행하지 않았다.

서버 endpoint는 /health, /v1/models, /v1/chat/completions다.
응답 content에는 content/organization/expression 각각의 정수 score와 rationale을 담은 JSON을 반환한다.
