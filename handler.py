"""RunPod Serverless Handler voor Hunyuan3D-2.1 generatie.

Accepteert een base64-encoded afbeelding, genereert een 3D mesh,
en retourneert het GLB bestand als base64.

Input:
  {
    "input": {
      "image_base64": "...",      # base64 encoded PNG/JPG
      "seed": 42,                  # optioneel
      "num_inference_steps": 25,   # optioneel (default: 25)
      "octree_resolution": 256     # optioneel
    }
  }

Output:
  {
    "glb_base64": "...",           # base64 encoded GLB
    "faces": 12345,
    "vertices": 6789,
    "duration_seconds": 15.2
  }
"""

import runpod
import base64
import io
import os
import sys
import time
import tempfile

# Voeg Hunyuan3D toe aan path
# Officiele repo: package = hy3dgen, import = hy3dgen.shapegen.pipelines
sys.path.insert(0, "/opt/hunyuan3d")

# Cache de pipeline globaal (persistent tussen requests op dezelfde worker)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    # Download model als het nog niet in cache staat
    cache_dir = os.environ.get("HY3DGEN_MODELS", "/opt/models")
    model_local = os.path.join(cache_dir, "tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/config.yaml")

    if not os.path.exists(model_local):
        print("[handler] Model niet in cache, downloaden via HuggingFace...")
        from huggingface_hub import hf_hub_download
        for fname in ["hunyuan3d-dit-v2-1/model.fp16.ckpt", "hunyuan3d-dit-v2-1/config.yaml"]:
            hf_hub_download("tencent/Hunyuan3D-2.1", fname,
                           local_dir=os.path.join(cache_dir, "tencent/Hunyuan3D-2.1"))
        print("[handler] Model gedownload!")

    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        device="cuda",
        dtype=torch.float16,
    )
    return _pipeline


def handler(job):
    """RunPod serverless handler."""
    job_input = job["input"]

    image_b64 = job_input.get("image_base64")
    if not image_b64:
        return {"error": "image_base64 is verplicht"}

    seed = job_input.get("seed", 42)
    steps = job_input.get("num_inference_steps", 25)
    octree_res = job_input.get("octree_resolution", 256)

    import torch
    import numpy as np
    from PIL import Image

    # Decode afbeelding
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Achtergrond verwijderen
    try:
        from hy3dgen.shapegen.rembg import BackgroundRemover
        rembg = BackgroundRemover()
        image = rembg(image)
    except Exception:
        pass

    # Genereer
    pipeline = get_pipeline()

    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    mesh = pipeline(
        image=image,
        num_inference_steps=steps,
        octree_resolution=octree_res,
    )[0]
    duration = time.time() - t0

    # Exporteer naar GLB bytes
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=True) as f:
        mesh.export(f.name)
        f.seek(0)
        glb_bytes = open(f.name, "rb").read()

    glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")

    return {
        "glb_base64": glb_b64,
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "duration_seconds": round(duration, 1),
        "seed": seed,
    }


runpod.serverless.start({"handler": handler})
