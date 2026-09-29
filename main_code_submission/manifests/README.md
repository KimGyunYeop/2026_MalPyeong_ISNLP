# 제출 manifest

이 폴더의 JSON은 어떤 score checkpoint와 rationale adapter를 한 Docker image에 넣을지 정하는
host-side source manifest다. 연구 default나 결과 폴더의 최신 시각으로 모델을 자동 선택하지 않는다.
현재 score는 `c02_avg_mse025/a228dbe60ed8`, rationale는 Gemma teacher 11,600편으로 증류한 A.X
LoRA로 확정·배포됐다. 이 폴더에는 역사 Y6 재현용 `Y6_matched_fallback.json`만 보존한다. 현행
source manifest는 루트 `docker_release.sh`가 고정 artifact와 prompt closure에서 결정론적으로 만들며
[`results/docker_release/source_manifests/`](../results/docker_release/source_manifests/)에 보관한다.

## Source와 staged manifest

- source manifest: host의 checkpoint와 adapter 경로, 모델 revision, release provenance를 가진다.
- staged manifest: [`build_image.sh`](../build_image.sh)가 source asset을 image 안으로 복사한 뒤
  `/opt/submission` 기준 경로와 실제 deployed artifact fingerprint로 다시 만든다.

Staged manifest를 손으로 편집하거나 이전 build의 `submission_assets/`를 다음 release의 source로
사용하지 않는다.

## 최종 선택에 필요한 정보

- `score_members`: 이름, 선택 checkpoint, backbone key와 weight
- `rationale`: base model/revision, adapter, shared-backbone key, prompt 원문·ID·SHA-256,
  chat-template SHA-256, 길이와 dtype/load 계약
- top level: served model ID, essay surface, `score_postprocess`, generation seed와 token 제한
- `extra`: candidate/config ID, checkpoint selector·step, source artifact SHA-256, 고정 image tag와
  release revision

Score와 rationale가 같은 backbone을 공유할 때도 두 adapter의 rank와 target module, key mapping,
artifact fingerprint를 각각 확인한다. 모델이나 CUDA load 실패를 CPU, base-only, 중립 점수 또는
기본 근거로 대체하는 설정은 두지 않는다.

## 갱신 순서

1. score와 rationale artifact를 명시한다.
2. `docker_release.sh check`로 source artifact와 prompt closure를 다시 계산해 manifest에 기록한다.
3. 새 고정 tag를 사용해 image를 build한다.
4. 같은 image ID로 no-argument 기동, 세 endpoint와 validation HTTP 전건을 검사한다.
5. 성공한 image만 push하고 registry digest를 다른 서버에서 다시 평가한다.

현행 사용자 진입점은 루트 [`docker_release.sh`](../../docker_release.sh), backend는
[`release_final.sh`](../release_final.sh)이다. 현재 운영 절차는 ``README.md``, 요청·제출 규정은
``SUBMISSION_FORM_AND_EVAL_GUIDE_20260810.md``를
따른다. 과거 Y6 예측을 재현할 때만
``Y6_PREDICTION_PARITY_CONTRACT.md``의 historical oracle을
사용한다.
