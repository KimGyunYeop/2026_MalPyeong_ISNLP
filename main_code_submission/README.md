# 제출 Docker — 현행 최소 경로

> **[2026-08-25 정정]** 이 문서가 "확정"이라 적은 채점 모델(c02 / `skt/A.X-4.0-Light`)은
> 최종 제출본이 **아니다**. 최종 제출본은 `Qwen/Qwen3.5-9B` 8 seed 앙상블이며 근거 모델도
> Qwen student(v4 prompt)다. 아래 내용은 그 결정에 이르는 과정의 기록으로 읽을 것.
> 현재 확정 구성은 저장소 루트 ``README.md``와 `python show_submission_settings.py`가 기준이다.

현재 상태의 단일 기준은 ``CURRENT_RELEASE_STATUS.md``다. score는
`c02_avg_mse025/a228dbe60ed8`, rationale는 Gemma teacher 11,600편으로 증류한 A.X LoRA로 확정됐고
두 모델을 묶은 image의 local 400편 검증과 public registry push까지 완료했다. 외부 단일 L40S
anonymous-pull 검증만 남아 있다.

[`manifests/Y6_matched_fallback.json`](manifests/Y6_matched_fallback.json)과 `release_y6.sh`는 과거
Y6/r6 재현용이다. 새 최종 제출에 사용하지 않는다.

```text
docker run --network eval-net --gpus all IMAGE
  → CUDA BF16 확인
  → 확정 score model + rationale adapter 동기 로드
  → 0.0.0.0:8000에서 HTTP 서버 시작
  → /health, /v1/models, /v1/chat/completions
```

모델 로드가 실패하면 traceback을 남기고 컨테이너가 종료된다. CPU, 일부 모델, 미학습 base,
중립 3점, 기본 근거로 대체하지 않는다. 추론 한 건이 실패하면 그 요청이 HTTP 500이 되며 가짜
점수는 반환하지 않는다.

## 최종 확정 구성

- score: c02 LoRA + score heads, checkpoint selector `best_checkpoint_official_matched_rmse`
- rationale: Gemma teacher 11,600편으로 증류한 A.X BF16 LoRA r32/a64, step 726
- image tag: `c02-gemma-full11600-cu128-offline-r1-20260812`
- image/digest: `sha256:5b4dd17114debf44a68d480775f90b1aa648185efa5b8de431745e9a9588f7e3`
- HTTP 400편 score oracle: `ca9b4244e95b0d1694d1aa85bd68194c0f1140a23fa2eee34f8f33c2e94047a7`
- base: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`
- weights: image에 내장한 baked mode만
- model: A.X-4.0-Light backbone 한 벌
- postprocess: 학습·로컬 검증과 같은 `average_matched`, 1~5 정수
- offline cache: 실행 uid/HOME과 무관한 `/opt/submission/hf/hub`

`average_matched`는 세 연속 점수를 1~5 정수로 만들면서 정수 합이 연속 점수 합의 사사오입과
일치하도록 잔여를 배분한다. Docker가 반환하는 값은 이미 정수이므로 평가기의 영역별 사사오입은
그 값에 대해 항등 연산이다. 따라서 score 실험에서 선택한 제출 표면을 그대로 유지한다.

## 입력과 출력

평가 요청은 system role 없이 user message 하나다.

```text
[채점 지시사항]

[prompt_text]
주제

[essay_text]
에세이
```

`request_parser.py`는 마지막 `[prompt_text]`, `[essay_text]`를 기준으로 주제와 본문을 원문 그대로
추출한다. 응답의 `choices[0].message.content`에는 wrapper나 Markdown 없이 다음 JSON 하나만 넣는다.

```json
{
  "content": {"score": 4, "rationale": "..."},
  "organization": {"score": 3, "rationale": "..."},
  "expression": {"score": 4, "rationale": "..."}
}
```

마커 없는 Docker 규정의 일반 인사 smoke에는 C/O/E 점수를 꾸며 내지 않고, OpenAI envelope 안에
사용법 안내 문자열을 반환한다.

## 파일 역할

- `serve.py`: CUDA 확인, 동기 모델 로드, 필수 세 endpoint
- `engine.py`: 점수 추론, `average_matched` 정수화, rationale adapter 전환/생성
- `request_parser.py`: 공식 단일-user 입력 분리
- `schema.py`: 공식 parser 복제와 안전한 JSON 조립
- `config.py`: 고정 asset 경로와 load 계약
- `Dockerfile`: cu12.8, baked weights, 인자 없는 entrypoint
- `build_image.sh`: manifest asset staging과 image build

## 빌드·검증·업로드

현행 사용자 진입점은 루트의 [`docker_release.sh`](../docker_release.sh)다.
`release_final.sh`은 이 wrapper의 backend이고 `release_y6.sh`은 역사 Y6 재현용이다. 이미 push된
기본 tag는 원격 overwrite가 차단되므로 아래 `push`를 그대로 다시 실행하지 않는다.

```bash
# 고정 artifact, prompt와 manifest preflight만
bash docker_release.sh check

# build + HTTP validation 400편, push 없음
GPU=all bash docker_release.sh verify

# 새 immutable tag를 발급할 때만 build + HTTP validation 400편 + push
IMAGE_TAG=<새-tag> REGISTRY_REPO=gyunyeop/writing-scorer GPU=all \
  bash docker_release.sh push
```

wrapper와 backend는 다음 순서를 강제한다.

1. manifest의 새 고정 tag로 build
2. 같은 image ID를 UID 1000·`HOME=/tmp`·무인터넷에서 offline asset 검사
3. 같은 image ID를 추가 인자 없이 GPU에서 실행하고 health/models/OpenAI 요청과 validation
   400편 검사
4. 모두 성공했을 때만 push
5. `release.txt`에 manifest/prediction hash, RepoDigest와 제출 URL 기록

400편 검사에는 validation 순서의 `essay_id/C/O/E 정수` SHA-256을 **해당 final manifest에 고정한
HTTP oracle**과 exact 비교하는 gate가 포함된다. 현행 hash는 `ca9b4244…047a7`이고 역사적 Y6
oracle `47e2aad8…14c72`는 Y6 재현 시에만 사용한다.

`verify`와 `push`는 각각 독립 build·400편 검사를 수행하므로 연달아 실행하면 평가가 두 번이다.
이번 release도 `verify`를 먼저 별도로 실행한 뒤 `push`가 이전 image를 재사용하지 않고 다시
build·full-400 검증하는 fail-closed 설계 때문에 두 번 평가됐다. 새 release는 fresh 상태에서
`push`만 실행하면 build+full-400+push를 한 번에 수행한다. push 성공의 권위 metadata는
[`release.txt`](results/docker_release/c02-gemma-full11600-cu128-offline-r1-20260812_20260812_153427/release.txt)다.

## 다른 서버 확인

`code_for_docker_check_otherserv/` 전체를 다른 L40S 서버에 복사하고 release 로그의 digest로 실행한다.

```bash
cd code_for_docker_check_otherserv
bash check_uploaded_final.sh /PATH/TO/PUSH_SUCCESS/release.txt
```

이 검사가 anonymous pull, `--gpus all` no-argument 기동, 세 endpoint, 긴 입력, 400편을 모두
통과하기 전까지 외부 환경 승인은 미완료다. 통과한 뒤 제출 폼에는 `release.txt`의
`submission_url`인
`docker://gyunyeop/writing-scorer:c02-gemma-full11600-cu128-offline-r1-20260812`를 그대로 넣는다.
예시 tag를 손으로 조립하지 않는다.
과거 `y6-matched-fallback-20260811`, `healthfix-r2`, `failfast-r4`, `minimal-r5`, `offline-r6`
tag는 재사용하지 않는다.
