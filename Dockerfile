# CUDA runtime base with PyTorch preinstalled — matches the cu12.x stack the
# project trains on. This image includes torch; we add the serving deps.
# (Ada/Blackwell GPUs work with the 2.7+ / cu12.8 images.)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install Python deps first so this layer caches across code changes.
COPY requirements.txt serve/requirements.txt ./serve/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r serve/requirements.txt

# App code and the fine-tuned adapter. The adapter is only a few MB (LoRA),
# so baking it into the image is fine; the base model is pulled from the HF
# cache at startup (mount a volume to avoid re-downloading — see README).
COPY src ./src
COPY serve ./serve
COPY runs/qwen1.5b-lora/final ./runs/qwen1.5b-lora/final

ENV MODEL_NAME=Qwen/Qwen2.5-Coder-1.5B-Instruct \
    ADAPTER_PATH=runs/qwen1.5b-lora/final \
    INCLUDE_SAMPLES=1 \
    LOG_PATH=logs/requests.jsonl \
    HF_HOME=/app/.hf_cache

EXPOSE 8000

# One worker: the model is loaded once per process and holds the GPU; scaling
# is horizontal (more containers), not more workers in one container.
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
