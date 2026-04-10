FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HY3DGEN_MODELS=/opt/models

# RunPod serverless SDK + mesh tools
RUN pip install --no-cache-dir runpod trimesh fast-simplification pillow \
    diffusers transformers accelerate huggingface-hub

# Clone Hunyuan3D
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/hunyuan3d

# Installeer Hunyuan3D
RUN cd /opt/hunyuan3d && pip install --no-cache-dir -e . || pip install --no-cache-dir -r requirements.txt || true

# Download model via wget (betrouwbaarder dan hf_hub_download in Docker build)
RUN mkdir -p /opt/models/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1 && \
    wget -q -O /opt/models/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/config.yaml \
      "https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/hunyuan3d-dit-v2-1/config.yaml" && \
    wget -O /opt/models/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt \
      "https://huggingface.co/tencent/Hunyuan3D-2.1/resolve/main/hunyuan3d-dit-v2-1/model.fp16.ckpt" && \
    ls -lh /opt/models/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/

# Kopieer onze worker
COPY handler.py /opt/handler.py

CMD ["python", "/opt/handler.py"]
