FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

# RunPod serverless SDK + mesh tools
RUN pip install --no-cache-dir runpod trimesh fast-simplification pillow \
    diffusers transformers accelerate huggingface-hub hf-xet

# Clone Hunyuan3D
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/hunyuan3d

# Installeer Hunyuan3D als package
RUN cd /opt/hunyuan3d && pip install --no-cache-dir -e . || pip install --no-cache-dir -r requirements.txt || true

# Pre-download model weights naar de HY3DGEN_MODELS cache locatie
ENV HY3DGEN_MODELS=/opt/models
RUN python -c "import hf_xet; from huggingface_hub import hf_hub_download; hf_hub_download('tencent/Hunyuan3D-2.1', 'hunyuan3d-dit-v2-1/model.fp16.ckpt', local_dir='/opt/models/tencent/Hunyuan3D-2.1'); hf_hub_download('tencent/Hunyuan3D-2.1', 'hunyuan3d-dit-v2-1/config.yaml', local_dir='/opt/models/tencent/Hunyuan3D-2.1')"

# Kopieer onze worker
COPY handler.py /opt/handler.py

CMD ["python", "/opt/handler.py"]
