import os
import shutil
import tempfile
import zipfile
import base64
from contextlib import asynccontextmanager

import torch
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete, Compose, EnsureType, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd,
    ConcatItemsd, ToTensord, ResizeWithPadOrCropd
)

from .processing import get_model

# --- Globals ---
model_cache = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... (lifespan function remains the same) ...
    print("Loading model...")
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
    model_cache["model"] = get_model(model_path, device)
    print("Model loaded successfully.")
    yield
    model_cache.clear()
    print("Cleaned up model cache.")

app = FastAPI(lifespan=lifespan)

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/segment-and-analyze/")
async def segment_and_analyze(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        # --- 1 & 2. Data Ingestion & Conversion ---
        # ... (This part is correct and remains unchanged) ...
        zip_path = os.path.join(temp_dir, file.filename)
        dicom_base_dir = os.path.join(temp_dir, "dicom_files")
        os.makedirs(dicom_base_dir)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dicom_base_dir)
        print(f"Extracted {file.filename}.")
        
        modality_map = {"FLAIR": "flair", "T1w": "t1", "T1wCE": "t1ce", "T2w": "t2"}
        nifti_paths_dict = {}
        voxel_volume = None
        
        for folder_name, modality_key in modality_map.items():
            modality_path = os.path.join(dicom_base_dir, folder_name)
            if not os.path.isdir(modality_path): continue
            
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(modality_path)
            if not dicom_names: continue
            
            reader.SetFileNames(dicom_names)
            sitk_image = reader.Execute()

            if voxel_volume is None:
                spacing = sitk_image.GetSpacing()
                voxel_volume = spacing[0] * spacing[1] * spacing[2]
            
            nifti_path = os.path.join(temp_dir, f"{modality_key}.nii.gz")
            sitk.WriteImage(sitk_image, nifti_path)
            nifti_paths_dict[modality_key] = nifti_path
        
        print("Converted DICOM series to NIfTI:", list(nifti_paths_dict.keys()))

        # --- 3. Preprocessing ---
        # ... (This part is correct and remains unchanged) ...
        required_modalities = ["flair", "t1", "t1ce", "t2"]
        if not all(m in nifti_paths_dict for m in required_modalities):
            raise ValueError("Missing one or more required modalities.")

        input_dict = {f"image_{m}": nifti_paths_dict[m] for m in required_modalities}
        image_keys = list(input_dict.keys())
        uniform_spatial_size = (240, 240, 155)
        
        transforms = Compose([
            LoadImaged(keys=image_keys),
            EnsureChannelFirstd(keys=image_keys),
            Orientationd(keys=image_keys, axcodes="RAS"),
            Spacingd(keys=image_keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ResizeWithPadOrCropd(keys=image_keys, spatial_size=uniform_spatial_size, method="symmetric", mode="constant"),
            ScaleIntensityRanged(keys=image_keys, a_min=0, a_max=4000, b_min=0.0, b_max=1.0, clip=True),
            ConcatItemsd(keys=image_keys, name="image", dim=0),
            ToTensord(keys=["image"]),
        ])

        processed_data = transforms(input_dict)
        input_tensor = processed_data['image'].to(device).unsqueeze(0)
        print("Image preprocessed. Final tensor shape:", input_tensor.shape)

        # --- 4. Inference ---
        model = model_cache["model"]
        with torch.no_grad(), torch.amp.autocast(device_type=str(device)):
            val_outputs = sliding_window_inference(input_tensor, (96, 96, 96), 4, model)
        
        seg_mask = torch.argmax(val_outputs, dim=1).squeeze(0).cpu().numpy()
        print("Inference complete.")

        # --- 5. CALCULATE VOLUME & CREATE MESH ---
        tumor_voxel_count = np.sum(seg_mask > 0)
        tumor_volume_mm3 = tumor_voxel_count * voxel_volume
        print(f"Calculated Volume: {tumor_volume_mm3:.2f} mm^3")

        print("Creating initial mesh from segmentation mask...")
        vertices, faces, _, _ = marching_cubes(seg_mask, level=0.5, spacing=(1.0, 1.0, 1.0))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(f"Initial mesh created with {len(mesh.faces)} faces.")

        # --- 6. (OPTIONAL) SAVE THE ORIGINAL, LARGE MESH FOR DEBUGGING ---
        debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_output')
        os.makedirs(debug_dir, exist_ok=True)
        original_mesh_path = os.path.join(debug_dir, "original_mesh.glb")
        mesh.export(original_mesh_path)
        print(f"DEBUG: Saved ORIGINAL mesh to {original_mesh_path}")

        # --- 7. SIMPLIFY THE MESH (USING THE CORRECT FUNCTION NAME) ---
        target_face_count = 15000
        simplified_mesh = mesh.copy() # Make a copy to simplify
        if len(simplified_mesh.faces) > target_face_count:
            print(f"Simplifying mesh from {len(simplified_mesh.faces)} to ~{target_face_count} faces...")
            simplified_mesh = simplified_mesh.simplify_quadric_decimation(target_face_count)
            print(f"Mesh simplified to {len(simplified_mesh.faces)} faces.")

        # --- 8. EXPORT AND ENCODE SIMPLIFIED MESH ---
        glb_data = trimesh.exchange.gltf.export_glb(simplified_mesh)
        glb_base64 = base64.b64encode(glb_data).decode('utf-8')
        print("Simplified mesh exported to GLB and encoded.")
        
        # --- (OPTIONAL) SAVE THE SIMPLIFIED MESH FOR DEBUGGING ---
        simplified_mesh_path = os.path.join(debug_dir, "simplified_mesh.glb")
        simplified_mesh.export(simplified_mesh_path)
        print(f"DEBUG: Saved SIMPLIFIED mesh to {simplified_mesh_path}")

        # --- 9. RETURN JSON RESPONSE ---
        return JSONResponse(content={
            "volume_mm3": tumor_volume_mm3,
            "model_glb_base64": glb_base64,
            "message": "Segmentation successful."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)