"""RunPod Serverless Handler voor Hunyuan3D-2.1 generatie."""

import runpod
import base64
import io
import os
import sys
import time
import tempfile

# Voeg Hunyuan3D toe aan path
sys.path.insert(0, "/opt/hunyuan3d")

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch

    # Stap 1: Download model naar HY3DGEN_MODELS cache
    cache_dir = os.environ.get("HY3DGEN_MODELS", os.path.expanduser("~/.cache/hy3dgen"))
    model_dir = os.path.join(cache_dir, "tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1")
    ckpt_path = os.path.join(model_dir, "model.fp16.ckpt")

    if not os.path.exists(ckpt_path):
        print(f"[handler] Model downloaden naar {cache_dir}...")
        import subprocess

        clone_dir = os.path.join(cache_dir, "tencent/Hunyuan3D-2.1")
        if not os.path.exists(os.path.join(clone_dir, ".git")):
            os.makedirs(cache_dir, exist_ok=True)
            # Sparse clone + LFS pull voor alleen het model bestand
            subprocess.run([
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "https://huggingface.co/tencent/Hunyuan3D-2.1", clone_dir
            ], check=True)
            subprocess.run(["git", "sparse-checkout", "set", "hunyuan3d-dit-v2-1"], cwd=clone_dir, check=True)
            subprocess.run(["git", "lfs", "pull", "--include", "hunyuan3d-dit-v2-1/*"], cwd=clone_dir, check=True)

        # Debug: toon wat er in de directory staat
        import subprocess
        subprocess.run(["find", clone_dir, "-type", "f", "-name", "*.ckpt", "-o", "-name", "*.yaml"], check=False)

        if os.path.exists(ckpt_path):
            print(f"[handler] Model gedownload: {os.path.getsize(ckpt_path)/(1024**3):.1f} GB")
        else:
            # Misschien staat het ergens anders?
            for root, dirs, files in os.walk(clone_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    print(f"[handler] Found: {fp} ({os.path.getsize(fp)} bytes)")
            raise FileNotFoundError(f"Model download mislukt: {ckpt_path}")

    # Stap 2: Laad pipeline
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        device="cuda",
        dtype=torch.float16,
    )
    print("[handler] Pipeline geladen op CUDA")
    return _pipeline


def handler(job):
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

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    try:
        from hy3dgen.shapegen.rembg import BackgroundRemover
        rembg = BackgroundRemover()
        image = rembg(image)
    except Exception:
        pass

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

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        tmp_path = f.name
    mesh.export(tmp_path)
    with open(tmp_path, "rb") as f:
        glb_bytes = f.read()
    os.unlink(tmp_path)

    glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")

    return {
        "glb_base64": glb_b64,
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "duration_seconds": round(duration, 1),
        "seed": seed,
    }


runpod.serverless.start({"handler": handler})
