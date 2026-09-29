# 제출 이미지는 한 가지 CUDA/runtime과 baked weights만 사용한다. build argument로 다른
# runtime을 같은 tag에 넣는 이중 경로를 두지 않는다.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# 모델/CUDA/adapter는 정확히 한 번 로드한다. 일부 구성원, CPU, base-only 근거 모델로
# 후퇴하거나 중립 점수를 반환하는 환경변수 우회는 두지 않는다.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/submission/hf \
    HF_HUB_CACHE=/opt/submission/hf/hub \
    HUGGINGFACE_HUB_CACHE=/opt/submission/hf/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SUBMISSION_MANIFEST=/opt/submission/submission_manifest.json

WORKDIR /opt/app

COPY main_code_submission/requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# --- layer 순서 계약 -----------------------------------------------------------
# 아래는 "바뀌는 빈도가 낮은 것부터" 정렬한다. Docker layer cache는 첫 변경 지점부터
# 뒤를 전부 무효화하므로, 자주 고치는 application code가 앞에 있으면 코드 한 줄만
# 바꿔도 14 GiB짜리 HF cache 8개 layer를 다시 만든다. 실제로 그 순서였고, 디스크가
# 빠듯한 상태에서 재빌드할 때마다 40 GiB 가까이 새로 썼다. 그래서 application code를
# 맨 뒤로 옮긴다. 경로가 겹치지 않으므로(/opt/app 대 /opt/submission) 최종 파일
# 시스템은 순서와 무관하게 동일하다.
#
# GHCR은 단일 layer 10GB를 허용하지 않는다. build_image.sh는 HF cache를 snapshot symlink와
# 대상 blob이 함께 있는 8개 고정 source로 나누고 각 layer를 8GiB 미만으로 검사한다.
COPY submission_assets/hf_parts/part00/ /opt/submission/hf/hub/
COPY submission_assets/hf_parts/part01/ /opt/submission/hf/hub/
COPY submission_assets/hf_parts/part02/ /opt/submission/hf/hub/
COPY submission_assets/hf_parts/part03/ /opt/submission/hf/hub/
COPY submission_assets/hf_parts/part04/ /opt/submission/hf/hub/
COPY submission_assets/hf_parts/part05/ /opt/submission/hf/hub/
COPY submission_assets/hf_parts/part06/ /opt/submission/hf/hub/
COPY submission_assets/hf_parts/part07/ /opt/submission/hf/hub/
COPY submission_assets/hf_partition_manifest.json /opt/submission/hf_partition_manifest.json

# rationale는 두 score 후보가 공유한다. candidate-specific score checkpoint는 그 다음이다.
COPY submission_assets/rationale_adapter/ /opt/submission/rationale_adapter/
COPY submission_assets/checkpoints/ /opt/submission/checkpoints/

# PEFT 0.19의 `_create_new_module`은 8bit 사용 여부와 무관하게 `dispatch_bnb_8bit`를
# import하고, 그것이 bitsandbytes -> triton을 끌어온다. triton은 import 시점에 자체
# `driver.c`를 JIT 컴파일하므로 런타임에 C 컴파일러가 없으면 rationale adapter를 올리다가
# 컨테이너가 죽는다(2026-08-16 실측: "Failed to find C compiler").
# base image 태그가 갱신되며 컴파일러가 빠졌으므로 여기서 명시적으로 보장한다.
# 이 단계는 network가 필요하므로 자주 바뀌는 application code보다 **앞**에 둔다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cc --version

# --- 여기부터는 자주 바뀐다 (수 백 KB) ------------------------------------------
# 점수 계산은 main_code의 검증된 런타임 경로만 그대로 쓴다. 명시 목록과 root
# .dockerignore를 함께 유지해 results/datasets가 다시 build context로 들어오지 않게 한다.
# `main_code/utils.py`가 `official_metrics`를 import하므로 함께 넣는다. 빠지면
# container가 모델 로드 전에 ModuleNotFoundError로 죽는다(2026-08-20 빌드에서 실측).
COPY main_code/__init__.py main_code/config.py main_code/datasets.py \
     main_code/models.py main_code/official_metrics.py \
     main_code/paragraph_boundary.py main_code/postprocess.py \
     main_code/quantization_objectives.py main_code/utils.py \
     /opt/app/main_code/
COPY main_code/candidiate_model_list/all_models.tsv \
     /opt/app/main_code/candidiate_model_list/all_models.tsv
# 근거 프롬프트는 학습 코드를 직접 재사용하므로 이 세 파일이 런타임 의존성이다.
COPY main_code_relonation/__init__.py main_code_relonation/prompts.py \
     main_code_relonation/schema.py /opt/app/main_code_relonation/
COPY main_code_relonation/prompts/ /opt/app/main_code_relonation/prompts/

COPY main_code_submission/__init__.py main_code_submission/artifact_integrity.py \
     main_code_submission/config.py main_code_submission/degrade.py \
     main_code_submission/engine.py main_code_submission/prompts.py \
     main_code_submission/rationale_prompt_binding.py \
     main_code_submission/request_parser.py main_code_submission/schema.py \
     main_code_submission/serve.py \
     /opt/app/main_code_submission/

# release마다 새 tag/지문이 들어가므로 가장 마지막이다.
COPY submission_assets/submission_manifest.json /opt/submission/submission_manifest.json

EXPOSE 8000
ENTRYPOINT ["python", "-m", "main_code_submission.serve"]
