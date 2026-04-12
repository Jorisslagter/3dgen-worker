"""RunPod Serverless Handler voor Hunyuan3D-2.1 generatie + texture painting."""

import runpod
import base64
import gc
import io
import os
import shutil
import subprocess
import sys
import time
import tempfile

# Voeg Hunyuan3D-2.1 paden toe
sys.path.insert(0, "/opt/hunyuan3d")
sys.path.insert(0, "/opt/hunyuan3d/hy3dshape")
sys.path.insert(0, "/opt/hunyuan3d/hy3dpaint")

_pipeline = None
_paint_pipeline = None


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

        clone_dir = os.path.join(cache_dir, "tencent/Hunyuan3D-2.1")
        if not os.path.exists(os.path.join(clone_dir, ".git")) or not os.path.exists(ckpt_path):
            os.makedirs(cache_dir, exist_ok=True)
            # Bij partial clone: opruimen en opnieuw beginnen
            if os.path.exists(clone_dir) and not os.path.exists(ckpt_path):
                shutil.rmtree(clone_dir, ignore_errors=True)
            # Sparse clone + LFS pull voor alleen het model bestand
            try:
                subprocess.run([
                    "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    "https://huggingface.co/tencent/Hunyuan3D-2.1", clone_dir
                ], check=True)
                subprocess.run(["git", "sparse-checkout", "set", "hunyuan3d-dit-v2-1"], cwd=clone_dir, check=True)
                subprocess.run(["git", "lfs", "pull", "--include", "hunyuan3d-dit-v2-1/*"], cwd=clone_dir, check=True)
            except subprocess.CalledProcessError:
                shutil.rmtree(clone_dir, ignore_errors=True)
                raise

        if os.path.exists(ckpt_path):
            print(f"[handler] Model gedownload: {os.path.getsize(ckpt_path)/(1024**3):.1f} GB")
        else:
            # Misschien staat het ergens anders?
            for root, dirs, files in os.walk(clone_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    print(f"[handler] Found: {fp} ({os.path.getsize(fp)} bytes)")
            raise FileNotFoundError(f"Model download mislukt: {ckpt_path}")

    # Stap 2: Laad pipeline via from_single_file (omzeilt smart_load_model)
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    config_path = os.path.join(model_dir, "config.yaml")
    print(f"[handler] Loading from: ckpt={ckpt_path}, config={config_path}")

    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        ckpt_path=ckpt_path,
        config_path=config_path,
        device="cuda",
        dtype=torch.float16,
    )
    print("[handler] Pipeline geladen op CUDA")
    return _pipeline


def get_paint_pipeline():
    """Laad de Hunyuan3D-Paint pipeline (lazy, cached na eerste aanroep).

    Raises ImportError als texture dependencies niet beschikbaar zijn
    (bijv. nvdiffrast compile faalde tijdens build).
    """
    global _paint_pipeline
    if _paint_pipeline is not None:
        return _paint_pipeline

    # bpy (Blender) wordt alleen gebruikt in convert_obj_to_glb; die
    # functie gebruiken we niet (we converteren zelf met trimesh).
    # Mock bpy zodat de import van textureGenPipeline niet faalt.
    try:
        import bpy  # noqa: F401
    except ImportError:
        from unittest.mock import MagicMock
        sys.modules["bpy"] = MagicMock()
        print("[handler] bpy gemockt (niet geinstalleerd)")

    # torchvision_fix: basicsr/realesrgan verwachten een oude torchvision API
    try:
        sys.path.insert(0, "/opt/hunyuan3d")
        from torchvision_fix import apply_fix
        apply_fix()
        print("[handler] torchvision_fix toegepast")
    except Exception as e:
        print(f"[handler] torchvision_fix niet beschikbaar: {e}")

    # Diagnose: probeer mesh_inpaint_processor expliciet te importeren
    import glob, traceback, platform
    diag = []
    diag.append(f"python={platform.python_version()}")
    sos = glob.glob("/opt/hunyuan3d/hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor*")
    diag.append(f"files={sos}")
    try:
        from DifferentiableRenderer import mesh_inpaint_processor as _mip
        diag.append(f"import_ok: {[x for x in dir(_mip) if not x.startswith('_')]}")
    except Exception as e:
        diag.append(f"import_err={type(e).__name__}: {e}")
        diag.append(traceback.format_exc()[-500:])
    _mesh_inpaint_diag = " | ".join(diag)
    print(f"[handler] {_mesh_inpaint_diag}")

    try:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    except ImportError as e:
        raise ImportError(
            f"Texture painting niet beschikbaar: {e}. "
            "Dit betekent dat nvdiffrast of custom_rasterizer niet "
            "succesvol is gecompileerd tijdens de image build."
        ) from e

    max_num_view = 6
    resolution = 512

    print("[handler] Paint pipeline laden...")
    config = Hunyuan3DPaintConfig(max_num_view, resolution)
    config.realesrgan_ckpt_path = "/opt/hunyuan3d/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    config.multiview_cfg_path = "/opt/hunyuan3d/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    config.custom_pipeline = "/opt/hunyuan3d/hy3dpaint/hunyuanpaintpbr"
    config.device = "cuda"
    _paint_pipeline = Hunyuan3DPaintPipeline(config)
    print("[handler] Paint pipeline geladen op CUDA")
    return _paint_pipeline


def paint_mesh(mesh_glb_path, image, paint_prompt=None):
    """Draai Hunyuan3D-Paint op een gegenereerd mesh.

    Args:
        mesh_glb_path: Pad naar het witte GLB mesh
        image: PIL Image (referentie afbeelding voor stijl)
        paint_prompt: Optionele prompt (niet gebruikt door pipeline,
                      maar beschikbaar voor toekomstige uitbreidingen)

    Returns:
        Pad naar het getextureerde GLB bestand
    """
    paint_pipeline = get_paint_pipeline()

    work_dir = tempfile.mkdtemp(prefix="hy3d_paint_")
    output_mesh_path = os.path.join(work_dir, "textured_mesh.obj")

    # De paint pipeline accepteert een image (PIL of pad) en mesh pad
    # Het doet intern: remesh -> UV unwrap -> multiview diffusion ->
    # super-resolution -> baking -> export
    # save_glb=False want convert_obj_to_glb vereist bpy (niet geinstalleerd).
    # We converteren zelf met trimesh na afloop.
    paint_pipeline(
        mesh_path=mesh_glb_path,
        image_path=image,
        output_mesh_path=output_mesh_path,
        use_remesh=True,
        save_glb=False,
    )

    if not os.path.exists(output_mesh_path):
        raise FileNotFoundError(
            f"Paint pipeline heeft geen OBJ geproduceerd: {output_mesh_path}"
        )

    # OBJ -> GLB via trimesh (met materialen/textures)
    import trimesh
    output_glb_path = output_mesh_path.replace(".obj", ".glb")
    textured = trimesh.load(output_mesh_path, force="mesh", process=False)
    textured.export(output_glb_path)
    print(f"[handler] GLB geconverteerd via trimesh: {output_glb_path}")

    return output_glb_path


def handler(job):
    job_input = job["input"]

    image_b64 = job_input.get("image_base64")
    if not image_b64:
        return {"error": "image_base64 is verplicht"}

    seed = job_input.get("seed", 42)
    steps = job_input.get("num_inference_steps", 25)
    octree_res = job_input.get("octree_resolution", 256)
    do_texture = job_input.get("texture", False)
    paint_prompt = job_input.get("paint_prompt")

    import torch
    import numpy as np
    from PIL import Image

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    try:
        from hy3dshape.rembg import BackgroundRemover
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
    mesh_duration = time.time() - t0

    # Exporteer wit mesh als GLB
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        white_mesh_path = f.name
    mesh.export(white_mesh_path)
    with open(white_mesh_path, "rb") as f:
        glb_bytes = f.read()

    glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")

    result = {
        "glb_base64": glb_b64,
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "duration_seconds": round(mesh_duration, 1),
        "seed": seed,
    }

    # Optioneel: texture painting
    if do_texture:
        try:
            t1 = time.time()

            # Flush VRAM van shape pipeline voor paint pipeline
            gc.collect()
            torch.cuda.empty_cache()

            # Maak een RGB referentie-afbeelding voor de paint pipeline
            ref_image = image.copy()
            if ref_image.mode == "RGBA":
                white_bg = Image.new("RGB", ref_image.size, (255, 255, 255))
                white_bg.paste(ref_image, mask=ref_image.getchannel("A"))
                ref_image = white_bg
            ref_image = ref_image.convert("RGB")

            textured_glb_path = paint_mesh(
                white_mesh_path, ref_image, paint_prompt
            )

            paint_duration = time.time() - t1

            with open(textured_glb_path, "rb") as f:
                textured_glb_bytes = f.read()

            result["textured_glb_base64"] = base64.b64encode(
                textured_glb_bytes
            ).decode("utf-8")
            result["paint_duration_seconds"] = round(paint_duration, 1)
            result["duration_seconds"] = round(mesh_duration + paint_duration, 1)

            # Cleanup texture temp bestanden
            try:
                shutil.rmtree(os.path.dirname(textured_glb_path), ignore_errors=True)
            except Exception:
                pass

        except Exception as e:
            # Texture painting faalde, maar wit mesh is wel beschikbaar
            import traceback
            tb = traceback.format_exc()
            diag = globals().get("_mesh_inpaint_diag", "")
            result["texture_error"] = f"{type(e).__name__}: {e}"
            result["texture_diag"] = diag
            result["texture_traceback"] = tb[-1500:]
            print(f"[handler] Texture painting mislukt: {e}\n{tb}")

    # Cleanup wit mesh temp bestand
    try:
        os.unlink(white_mesh_path)
    except Exception:
        pass

    return result


runpod.serverless.start({"handler": handler})
