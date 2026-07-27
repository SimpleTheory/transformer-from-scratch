FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/huggingface-cache

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    --break-system-packages \
    -r /app/requirements.txt

COPY src/ /app/src/

COPY docker_train.sh /usr/local/bin/train-transformer

RUN chmod +x /usr/local/bin/train-transformer

WORKDIR /workspace

ENTRYPOINT ["train-transformer"]