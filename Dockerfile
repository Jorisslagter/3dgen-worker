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

# Force CUDA arch list (geen GPU nodig tijdens build)
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# nvdiffrast (NVIDIA differentiable rasterizer, optioneel voor texture)
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git \
    || echo "WARN: nvdiffrast install failed, texture painting disabled"

# Clone Hunyuan3D-2.1 (broncode + hy3dpaint met texture painting)
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 /opt/hunyuan3d && \
    cd /opt/hunyuan3d && pip install --no-cache-dir -r requirements.txt || true

# Download RealESRGAN weights (voor texture super-resolution)
RUN mkdir -p /opt/hunyuan3d/hy3dpaint/ckpt && \
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -O /opt/hunyuan3d/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
    || echo "WARN: RealESRGAN download failed"

# Compileer custom_rasterizer CUDA extension (optioneel - als dit faalt
# werkt alleen mesh generatie, geen texture painting)
RUN cd /opt/hunyuan3d/hy3dpaint/custom_rasterizer && \
    pip install --no-cache-dir -e . || echo "WARN: custom_rasterizer compile failed, texture painting disabled"

# Compileer mesh_inpaint_processor C++ extension (optioneel)
RUN cd /opt/hunyuan3d/hy3dpaint/DifferentiableRenderer && \
    g++ -O3 -Wall -shared -std=c++11 -fPIC \
        $(python3 -m pybind11 --includes) \
        mesh_inpaint_processor.cpp \
        -o mesh_inpaint_processor$(python3-config --extension-suffix) \
    || echo "WARN: mesh_inpaint_processor compile failed"

COPY handler.py /opt/handler.py

# Model wordt bij eerste request gedownload
# Paint model (~7GB) wordt bij eerste texture request gedownload

CMD ["python", "/opt/handler.py"]
