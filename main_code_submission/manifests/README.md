# 제출 매니페스트

Qwen3.5-9B ABCD 8시드와 근거 어댑터, 등가중 평균, SMR, offset 0 구성이다.
채점·근거 설정은 `main_code/configs/report_abcd.json`과
`main_code_relonation/recipes/r18_report_qwen35.json`을 사용한다.

학습 완료 후 루트 docker_release.sh check를 실행하면 실제 설정·step·seed·prompt를 확인해
가중치 해시가 포함된 source manifest를 main_code_submission/results/report_manifests/에 만든다.
Prompt와 chat-template hash는 실제 어댑터 sidecar에서 읽는다.
생성 매니페스트는 실행 산출물이므로 Git에서 제외하며, 빈 템플릿을 별도로 올리지 않는다.

build_image.sh는 source asset을 image에 복사하고 /opt/submission 기준의 staged manifest를 만든다.
과거 Y6_matched_fallback.json은 A.X 모델의 역사 기록이다. 현재 Qwen 모델에 연결하지 않는다.
성능 평가와 배포는 별도 작업이다.

기술서와 현재 구현 사이의 step·warmup, 손실 수식, sampling 재시도 및
teacher/prompt 차이는 루트 [README](../../README.md)에 정리되어 있다.
상단 max_tokens=512는 API 요청 형식 값이며, 실제 근거 생성 상한은
rationale.max_new_tokens=2048이다.
