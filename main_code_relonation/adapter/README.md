# Standalone rationale Docker adapter 경로

이 디렉터리는 `main_code_relonation/Dockerfile`의 연구용 standalone image가 `/opt/models/adapter`로
복사하는 자리다. 현행 full distillation의 adapter는 `results/.../lora/final_adapter/`에 저장하며,
최종 대회 Docker는 [`main_code_submission/build_image.sh`](../../main_code_submission/build_image.sh)가
manifest에서 선택해 staging한다.

따라서 이 폴더의 내용을 최종 제출 adapter의 source of truth로 사용하지 않는다. Standalone smoke를
명시적으로 만들 때만 완전한 PEFT adapter closure를 넣는다.
