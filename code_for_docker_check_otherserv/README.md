# 다른 서버에서 최종 Docker 확인

이 폴더 전체를 다른 NVIDIA GPU 서버로 복사한다. 필요한 것은 Docker Engine, NVIDIA
Container Toolkit, Python 3, `curl`뿐이며 별도 Python package 설치는 필요 없다.

## 최종 release와 다른 서버 실행

새 release의 진입점은 루트 `docker_release.sh`다. 고정 score/rationale artifact와 prompt를
검사해 source manifest를 만들며, image tag와 HTTP 400편 oracle이 없거나 맞지 않으면 push 전에
중단한다. `main_code_submission/release_final.sh`은 이 wrapper의 backend다.

release가 출력한 `release.txt`를 이 폴더와 함께 다른 서버로 복사한다. 공식 validation 원본은
`DATA_FILE`로 지정하거나 `data/official_validation_400.jsonl`에 둔다.

```bash
cd /PATH/TO/code_for_docker_check_otherserv
nvidia-smi
docker info

DATA_FILE=/absolute/path/to/official_validation.jsonl \
  bash check_uploaded_final.sh /PATH/TO/release.txt
```

metadata 파일 대신 digest를 직접 전달할 수도 있지만, 이때는 확정 prediction hash를 두 번째
인자로 반드시 함께 넣는다.

```bash
bash check_uploaded_final.sh \
  USER/writing-scorer@sha256:<64자리-digest> \
  <64자리-prediction-sha256>
```

`docker info` 권한이 없으면 먼저 재로그인하거나 `newgrp docker`를 권장한다. 아직 현재 셸에 그룹이
반영되지 않은 경우 스크립트는 최소한의 `sg docker -c` 경로를 자동으로 사용한다.

최종 타 서버 검사는 평가 형태와 같은 `--gpus all`만 허용한다. 특정 GPU 진단은 공용
`run_docker_check.sh`에서만 가능하다. 스크립트는 빈 임시 Docker credential로 공개 pull하므로
registry repository가 Public이어야 한다.

## 이 명령이 확인하는 것

- GPU 기동 전에 네트워크 없이 UID 1000·`HOME=/tmp`에서 `/opt/submission/hf/hub`의
  config, tokenizer, weight index와 모든 shard를 읽을 수 있는지 확인
- image 뒤에 command나 argument를 붙이지 않은 `docker run`
- GPU에서 CUDA·확정 score model·rationale adapter load
- `0.0.0.0:8000` bind를 host port publish로 확인
- `GET /health` 200
- `GET /v1/models`의 `data[0].id`
- marker 없는 일반 요청의 OpenAI Chat Completions 응답 형식
- 공지처럼 채점 지시문·`[prompt_text]`·`[essay_text]`를 합친 단일 `user` message
- 같은 marker 형식의 12,000자 긴 에세이 1편과 공식 parser 통과
- 공지 parser를 적용한 validation 400편 전체
- raw HTTP 응답, 공식 점수, RMSE/Spearman 저장
- validation 순서의 `essay_id/C/O/E 정수` SHA-256이 release manifest의 확정 HTTP oracle과
  exact 일치

최종 모델은 학습·로컬 검증과 같은 `average_matched` 방식으로 C/O/E 정수를 반환한다. 평가기의
영역별 사사오입은 이미 정수인 출력에 적용되므로 값이 바뀌지 않는다. 모델이나 CUDA load가
실패하면 CPU·base model·중립 점수로 대체하지 않고 container가 종료되며 `container.log`에 오류가
남는다.

## 결과

```text
results/uploaded_final_YYYYMMDD_HHMMSS/
├── pull.log
├── container.log
├── health.json
├── models.json
├── generic_smoke.json
└── evaluation/
    ├── raw_http.jsonl
    ├── records.jsonl
    ├── metrics.json
    ├── prediction_hash.json
    ├── summary.json
    └── metadata.json
```

마지막에 아래 문구, exact digest와 400편 metric이 출력되어야 한다.

```text
PASS: anonymous pull + no-argument GPU server + validation 400/400
```

실패하면 제출하지 않고 터미널의 첫 오류와 `container.log`를 확인한다. 이 검사는 공개
validation과 공개 parser를 재현하지만 비공개 test data, 운영 timeout/concurrency, LLM Judge는
복제하지 않는다.

검사가 통과하면 제출 폼에는 digest가 아니라 `release.txt`의 `submission_url` 고정 tag URL을
넣는다. digest는 다른 서버 검증에만 사용한다.
