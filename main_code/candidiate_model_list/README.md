# Backbone 후보 목록

폴더명 `candidiate_model_list`는 기존 script 경로 호환 때문에 철자를 바꾸지 않는다.
`all_models.tsv`는 모델 탐색 당시의 Hugging Face ID, revision, 크기, 활성화 여부와 메모를 보존한
연구 카탈로그다. `enabled=1`은 다운로드/실험 후보 표시일 뿐 현재 default 또는 submission
승자를 의미하지 않는다.

현행 score 설정은 [`configs/confirmed_final.json`](../configs/confirmed_final.json), 실제 Docker
배포 artifact와 pinned base revision은 [`docker_release.sh`](../../docker_release.sh) 및
``CURRENT_RELEASE_STATUS.md``가 권위다.
카탈로그의 `main` revision을 그대로 byte-exact 재현 계약으로 사용하지 않는다.
