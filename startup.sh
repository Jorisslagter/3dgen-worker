#!/bin/bash
set -e

MARKER="/opt/.cuda_extensions_built"

if [ ! -f "$MARKER" ]; then
    echo "[startup] CUDA extensions compileren (eerste opstart)..."

    # nvdiffrast
    echo "[startup] nvdiffrast installeren..."
    pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

    # custom_rasterizer CUDA extension
    echo "[startup] custom_rasterizer compileren..."
    cd /opt/hunyuan3d/hy3dpaint/custom_rasterizer && pip install --no-cache-dir -e .

    # mesh_inpaint_processor C++ extension
    echo "[startup] mesh_inpaint_processor compileren..."
    cd /opt/hunyuan3d/hy3dpaint/DifferentiableRenderer && \
        g++ -O3 -Wall -shared -std=c++11 -fPIC \
            $(python3 -m pybind11 --includes) \
            mesh_inpaint_processor.cpp \
            -o mesh_inpaint_processor$(python3-config --extension-suffix)

    touch "$MARKER"
    echo "[startup] CUDA extensions klaar."
else
    echo "[startup] CUDA extensions al gecompileerd, skip."
fi

echo "[startup] Handler starten..."
exec python /opt/handler.py
