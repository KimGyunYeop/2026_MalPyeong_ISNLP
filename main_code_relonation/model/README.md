# 예약된 legacy 경로

이 디렉터리는 과거 standalone rationale Docker의 local model mount를 위해 남아 있다. 현행 student
학습, validation 및 최종 submission Docker는 여기의 파일을 읽지 않는다.

최종 image의 A.X snapshot은 [`main_code_submission/build_image.sh`](../../main_code_submission/build_image.sh)가
선택한 score/rationale manifest를 검증해 `submission_assets/`와 image 내부
`/opt/submission/hf/hub`에 staging한다. 이 폴더에 임의 weight를 복사해 최종 Docker를 만들지 않는다.
