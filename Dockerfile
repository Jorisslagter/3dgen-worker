FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HY3DGEN_MODELS=/tmp/hy3dgen_models

# Git LFS voor model download + RunPod SDK + mesh tools
RUN apt-get update && apt-get install -y git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod trimesh fast-simplification pillow \
    diffusers transformers accelerate huggingface-hub

# Clone en installeer Hunyuan3D
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/hunyuan3d && \
    cd /opt/hunyuan3d && pip install --no-cache-dir -e . || pip install --no-cache-dir -r requirements.txt || true

# Model wordt bij eerste request gedownload via huggingface_hub
# naar /runpod-volume/models (persistent) of ~/.cache/hy3dgen (ephemeral)
# Eerste request duurt ~2-3 min extra, daarna gecached

COPY handler.py /opt/handler.py

CMD ["python", "/opt/handler.py"]
