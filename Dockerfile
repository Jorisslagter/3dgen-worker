FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

# RunPod serverless SDK + mesh tools
RUN pip install --no-cache-dir runpod trimesh fast-simplification pillow \
    diffusers transformers accelerate huggingface-hub hf-xet

# Clone Hunyuan3D en installeer hy3dshape als package
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/hunyuan3d && \
    pip install --no-cache-dir -r /opt/hunyuan3d/hy3dshape/requirements.txt 2>/dev/null || true && \
    echo "/opt/hunyuan3d/hy3dshape" > /usr/local/lib/python3.11/dist-packages/hy3dshape.pth && \
    python -c "import sys; sys.path.insert(0,'/opt/hunyuan3d/hy3dshape'); from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline; print('hy3dshape import OK')"

# Pre-download model weights (baked in voor snelle cold start)
RUN python -c "import hf_xet; from huggingface_hub import hf_hub_download; hf_hub_download('tencent/Hunyuan3D-2.1', 'hunyuan3d-dit-v2-1/model.fp16.ckpt', local_dir='/opt/models/hunyuan3d-2.1'); hf_hub_download('tencent/Hunyuan3D-2.1', 'hunyuan3d-dit-v2-1/config.yaml', local_dir='/opt/models/hunyuan3d-2.1')"

# Kopieer onze worker
COPY handler.py /opt/handler.py

CMD ["python", "/opt/handler.py"]
