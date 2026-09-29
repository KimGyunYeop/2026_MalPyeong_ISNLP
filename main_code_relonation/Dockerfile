FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG USE_ADAPTER=0
ARG TRUST_REMOTE_CODE=0
ARG LOAD_IN_4BIT=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    MODEL_PATH=/opt/models/model \
    ADAPTER_PATH=/opt/models/adapter \
    USE_ADAPTER=${USE_ADAPTER} \
    TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE} \
    LOAD_IN_4BIT=${LOAD_IN_4BIT} \
    TORCH_DTYPE=bfloat16 \
    MAX_LENGTH=8192 \
    MAX_NEW_TOKENS=512 \
    SERVED_MODEL_NAME=malpyeong-rationale

WORKDIR /app

COPY requirements.txt /tmp/rationale-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/rationale-requirements.txt

# Build from main_code_relonation/ after placing a complete local HF snapshot in
# model/ and, for a LoRA image, the PEFT checkpoint in adapter/. Both directories
# are copied into the image so evaluation never depends on a network download.
COPY model/ /opt/models/model/
COPY adapter/ /opt/models/adapter/

RUN test -f /opt/models/model/config.json && \
    if [ "${USE_ADAPTER}" = "1" ]; then \
      test -f /opt/models/adapter/adapter_config.json; \
    fi

RUN mkdir -p /app/main_code_relonation
COPY __init__.py artifacts.py config.py modeling.py prompts.py \
     request_parser.py schema.py serve.py /app/main_code_relonation/
COPY prompts/ /app/main_code_relonation/prompts/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
