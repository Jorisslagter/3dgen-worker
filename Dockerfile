FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HY3DGEN_MODELS=/tmp/hy3dgen_models
# Nodig voor nvdiffrast
ENV CUDA_HOME=/usr/local/cuda
ENV EGL_PLATFORM=device

# Systeem dependencies: Git LFS, OpenGL (voor nvdiffrast), build tools
RUN apt-get update && apt-get install -y \
    git-lfs \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libegl1-mesa-dev \
    libglfw3-dev \
    pybind11-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies: RunPod SDK, mesh tools, texture painting deps
RUN pip install --no-cache-dir runpod trimesh fast-simplification pillow \
    diffusers transformers accelerate huggingface-hub \
    xatlas pybind11 scipy realesrgan basicsr

# Clone Hunyuan3D (broncode + hy3dpaint, zonder CUDA compilatie)
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/hunyuan3d && \
    cd /opt/hunyuan3d && pip install --no-cache-dir -e . || pip install --no-cache-dir -r requirements.txt || true

# nvdiffrast en CUDA extensions worden bij eerste opstart gecompileerd
# (vereist runtime GPU, kan niet tijdens docker build op macOS)

COPY handler.py /opt/handler.py
COPY startup.sh /opt/startup.sh
RUN chmod +x /opt/startup.sh

# Model wordt bij eerste request gedownload via huggingface_hub
# Paint model (~7GB) wordt bij eerste texture request gedownload

CMD ["/opt/startup.sh"]
