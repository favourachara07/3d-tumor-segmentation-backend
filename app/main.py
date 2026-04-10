"""
Nextar NeuroVista AI — v1 Backend
FastAPI endpoint: POST /segment-and-analyze/

Accepts : ZIP of 4 MRI DICOM series (FLAIR, T1w, T1wCE, T2w)
Returns :
  • clinical_report  – structured per-subregion volumes + derived metrics
  • model_glb_base64 – Base64-encoded GLB scene containing:
        - semi-transparent brain shell  (gray)
        - edema mesh                    (yellow)
        - necrotic core mesh            (dark-red)
        - enhancing tumour mesh         (bright red / hot)
"""

import os
import time
import shutil
import tempfile
import zipfile
import base64
from contextlib import asynccontextmanager

import torch
import numpy as np
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter           # ← organic smoothing

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from .processing import get_model, calculate_clinical_volumes


# ---------------------------------------------------------------------------
# Colour palette (RGBA 0-255) — BraTS clinical convention
# ---------------------------------------------------------------------------
COLOUR_BRAIN     = [160, 180, 200,  30]   # steel-blue, ghost-transparent shell
COLOUR_EDEMA     = [255, 215,   0, 210]   # gold / yellow
COLOUR_NECROTIC  = [160,  20,  20, 230]   # dark crimson
COLOUR_ENHANCING = [255,  50,  50, 255]   # bright red, fully opaque

# ---------------------------------------------------------------------------
# Gaussian smoothing sigmas (in voxels)
# Higher sigma = smoother organic look, but slightly softens the boundary.
# Tune these values if the surfaces look over- or under-smoothed.
#   Brain shell  : 1.2  — large structure, benefits most from rounding
#   Edema        : 0.9  — diffuse region, moderate smoothing
#   Necrotic     : 0.7  — smaller region, preserve detail
#   Enhancing    : 0.7  — clinically precise, minimal over-smoothing
# ---------------------------------------------------------------------------
SIGMA_BRAIN     = 1.2
SIGMA_EDEMA     = 0.9
SIGMA_NECROTIC  = 0.7
SIGMA_ENHANCING = 0.7


# ---------------------------------------------------------------------------
# Global model cache & device
# ---------------------------------------------------------------------------
model_cache: dict = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NeuroVista] Using device: {device}")


# ---------------------------------------------------------------------------
# App lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[NeuroVista] Loading segmentation model …")
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "best_model.pth"
    )
    model_cache["model"] = get_model(model_path, device)
    print("[NeuroVista] Model loaded successfully.")
    yield
    model_cache.clear()
    print("[NeuroVista] Model cache cleared.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Nextar NeuroVista AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Mesh helper  (Gaussian pre-smoothing for organic iso-surface)
# ---------------------------------------------------------------------------
def _build_mesh(
    binary_mask: np.ndarray,
    spacing: tuple,
    face_colour: list,
    sigma: float = 0.8,
    max_faces: int = 8_000,
) -> trimesh.Trimesh | None:
    """
    Build a smooth, coloured trimesh from a binary volumetric mask.

    Pipeline
    --------
    1. Gaussian blur the float-cast mask  →  rounds sharp voxel edges into
       smooth anatomical curves before marching_cubes runs.
       sigma=0 disables smoothing entirely (raw voxel edges).
    2. marching_cubes at iso-level 0.5  →  boundary sits on the original edge
    3. Quadric decimation to cap triangle count (keeps file size small)
    4. Per-face RGBA colour assignment

    Parameters
    ----------
    binary_mask : np.ndarray  bool or uint8, shape (H, W, D)
    spacing     : (sx, sy, sz) voxel size in mm — passed to marching_cubes
    face_colour : [R, G, B, A] 0-255
    sigma       : Gaussian σ in voxels. 0.7-1.2 is the sweet spot.
    max_faces   : target face count after decimation
    """
    if not np.any(binary_mask):
        return None

    try:
        # ── 1. Smooth ──────────────────────────────────────────────────────
        if sigma > 0:
            float_mask = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma)
        else:
            float_mask = binary_mask.astype(np.float32)

        # ── 2. Iso-surface extraction ───────────────────────────────────────
        verts, faces, _, _ = marching_cubes(
            float_mask, level=0.5, spacing=spacing
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # ── 3. Decimate ─────────────────────────────────────────────────────
        if len(mesh.faces) > max_faces:
            mesh = mesh.simplify_quadric_decimation(max_faces)

        # ── 4. Colour ───────────────────────────────────────────────────────
        rgba = np.array(face_colour, dtype=np.uint8)
        mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))

        return mesh

    except Exception as exc:
        print(f"[NeuroVista] Mesh build failed — {exc}; skipping layer.")
        return None


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@app.post("/segment-and-analyze/")
async def segment_and_analyze(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        # ------------------------------------------------------------------ #
        # 1. Ingest ZIP                                                        #
        # ------------------------------------------------------------------ #
        zip_path       = os.path.join(temp_dir, file.filename)
        dicom_base_dir = os.path.join(temp_dir, "dicom_files")
        os.makedirs(dicom_base_dir)

        with open(zip_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dicom_base_dir)
        print(f"[NeuroVista] Extracted {file.filename}.")

        # ------------------------------------------------------------------ #
        # 2. DICOM → NIfTI                                                     #
        # ------------------------------------------------------------------ #
        modality_map = {
            "FLAIR":  "flair",
            "T1w":    "t1",
            "T1wCE":  "t1ce",
            "T2w":    "t2",
        }
        nifti_paths: dict            = {}
        voxel_volume_mm3: float | None = None
        voxel_spacing: tuple         = (1.0, 1.0, 1.0)   # sensible default

        for folder_name, modality_key in modality_map.items():
            modality_path = os.path.join(dicom_base_dir, folder_name)
            if not os.path.isdir(modality_path):
                continue

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(modality_path)
            if not dicom_names:
                continue

            reader.SetFileNames(dicom_names)
            sitk_image = reader.Execute()

            if voxel_volume_mm3 is None:
                sx, sy, sz       = sitk_image.GetSpacing()
                voxel_volume_mm3 = sx * sy * sz
                voxel_spacing    = (sx, sy, sz)

            nifti_path = os.path.join(temp_dir, f"{modality_key}.nii.gz")
            sitk.WriteImage(sitk_image, nifti_path)
            nifti_paths[modality_key] = nifti_path

        print(f"[NeuroVista] Converted DICOM → NIfTI: {list(nifti_paths.keys())}")

        # ------------------------------------------------------------------ #
        # 3. Pre-processing                                                    #
        # ------------------------------------------------------------------ #
        required = ["flair", "t1", "t1ce", "t2"]
        missing  = [m for m in required if m not in nifti_paths]
        if missing:
            raise ValueError(f"Missing modalities: {missing}")

        input_dict   = {f"image_{m}": nifti_paths[m] for m in required}
        image_keys   = list(input_dict.keys())
        SPATIAL_SIZE = (240, 240, 155)

        transforms = Compose([
            LoadImaged(keys=image_keys),
            EnsureChannelFirstd(keys=image_keys),
            Orientationd(keys=image_keys, axcodes="RAS"),
            Spacingd(keys=image_keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ResizeWithPadOrCropd(
                keys=image_keys,
                spatial_size=SPATIAL_SIZE,
                method="symmetric",
                mode="constant",
            ),
            ScaleIntensityRanged(
                keys=image_keys,
                a_min=0, a_max=4000,
                b_min=0.0, b_max=1.0,
                clip=True,
            ),
            ConcatItemsd(keys=image_keys, name="image", dim=0),
            ToTensord(keys=["image"]),
        ])

        processed    = transforms(input_dict)
        input_tensor = processed["image"].to(device).unsqueeze(0)   # (1,4,H,W,D)
        print(f"[NeuroVista] Pre-processed tensor shape: {input_tensor.shape}")

        # ------------------------------------------------------------------ #
        # 4. Inference (timed)                                                #
        # ------------------------------------------------------------------ #
        model = model_cache["model"]
        t0    = time.perf_counter()

        with torch.no_grad():
            raw_output = sliding_window_inference(
                input_tensor, (96, 96, 96), 4, model
            )

        inference_time = round(time.perf_counter() - t0, 3)
        print(f"[NeuroVista] Inference done in {inference_time}s.")

        # Integer label mask  shape: (H, W, D)
        seg_mask   = torch.argmax(raw_output, dim=1).squeeze(0).cpu().numpy()

        # Brain structural shell from T1 channel (index 1)
        brain_mask = (input_tensor[0, 1].cpu().numpy() > 0.01).astype(np.uint8)

        # ------------------------------------------------------------------ #
        # 5. Volume calculation                                                #
        # ------------------------------------------------------------------ #
        volume_report = calculate_clinical_volumes(seg_mask, voxel_volume_mm3)
        volume_report["metadata"] = {
            "inference_time_seconds": inference_time,
            "warnings": volume_report.pop("warnings", []),
        }
        print(
            "[NeuroVista] Volumes — "
            f"Total: {volume_report['volumes_mm3']['total_tumor']:.1f} mm³  |  "
            f"Edema: {volume_report['volumes_mm3']['edema']:.1f} mm³  |  "
            f"Enhancing: {volume_report['volumes_mm3']['enhancing_tumor']:.1f} mm³"
        )

        # ------------------------------------------------------------------ #
        # 6. 3-D mesh generation  (Gaussian-smoothed for organic appearance)  #
        #                                                                      #
        #  Layer order: brain first (back) → tumour sub-regions on top (front)#
        # ------------------------------------------------------------------ #
        print("[NeuroVista] Building smoothed 3-D meshes …")

        brain_mesh     = _build_mesh(
            brain_mask,    voxel_spacing, COLOUR_BRAIN,
            sigma=SIGMA_BRAIN,     max_faces=12_000,
        )
        edema_mesh     = _build_mesh(
            seg_mask == 2, voxel_spacing, COLOUR_EDEMA,
            sigma=SIGMA_EDEMA,     max_faces=6_000,
        )
        necrotic_mesh  = _build_mesh(
            seg_mask == 1, voxel_spacing, COLOUR_NECROTIC,
            sigma=SIGMA_NECROTIC,  max_faces=4_000,
        )
        enhancing_mesh = _build_mesh(
            seg_mask == 3, voxel_spacing, COLOUR_ENHANCING,
            sigma=SIGMA_ENHANCING, max_faces=4_000,
        )

        # ------------------------------------------------------------------ #
        # 7. Compose GLB scene                                                #
        # ------------------------------------------------------------------ #
        scene = trimesh.Scene()

        for name, mesh in [
            ("brain",           brain_mesh),
            ("edema",           edema_mesh),
            ("necrotic_core",   necrotic_mesh),
            ("enhancing_tumor", enhancing_mesh),
        ]:
            if mesh is not None:
                scene.add_geometry(mesh, node_name=name)
                print(f"[NeuroVista]   + {name}: {len(mesh.faces)} faces")

        glb_bytes = trimesh.exchange.gltf.export_glb(scene)

        # Debug dump — remove for production deployment
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        safe_stem = os.path.splitext(os.path.basename(file.filename or "scene"))[0]
        out_path  = os.path.join(debug_dir, f"{safe_stem}_{int(time.time())}.glb")
        with open(out_path, "wb") as f:
            f.write(glb_bytes)
        print(f"[NeuroVista] Debug GLB saved → {out_path}")

        glb_base64 = base64.b64encode(glb_bytes).decode("utf-8")
        print("[NeuroVista] Scene encoded.")

        # ------------------------------------------------------------------ #
        # 8. Structured JSON response                                         #
        # ------------------------------------------------------------------ #
        return JSONResponse(
            content={
                "clinical_report": volume_report,
                "model_glb_base64": glb_base64,
                "message": "Segmentation successful.",
            }
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )

    finally:
        print(f"[NeuroVista] Cleaning up: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)