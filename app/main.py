"""
Nextar NeuroVista AI — v1 Backend  (spatial coherence fix)
"""

import os, time, shutil, tempfile, zipfile, base64
from contextlib import asynccontextmanager
from .clinical_decision import generate_clinical_report

import torch, numpy as np, SimpleITK as sitk, trimesh, mcubes
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as ndimage_label

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, ConcatItemsd, EnsureChannelFirstd, LoadImaged,
    Orientationd, ResizeWithPadOrCropd, ScaleIntensityRanged,
    Spacingd, ToTensord,
)
from .processing import get_model, calculate_clinical_volumes

COLOUR_BRAIN     = [160, 180, 200,  50]
COLOUR_EDEMA     = [255, 215,   0, 140]
COLOUR_NECROTIC  = [160,  20,  20, 180]
COLOUR_ENHANCING = [255,  50,  50, 200]

SIGMA_BRAIN=1.2; SIGMA_EDEMA=0.9; SIGMA_NECROTIC=0.7; SIGMA_ENHANCING=0.7

CONFIDENCE_THRESHOLDS = {
    1: 0.50,   # Necrotic Core  — accurate, standard threshold
    2: 0.50,   # Edema          — accurate, standard threshold
    # Enhancing: lowered 0.85 → 0.70 so the real enhancing core inside the
    # necrotic mass (moderate confidence) survives to the spatial coherence
    # filter. The ectopic hemisphere cluster is still removed by
    # _keep_spatially_coherent_components since it is >60 mm from edema.
    3: 0.70,
}

# Key constant: max allowed distance (mm) from edema centroid
# for necrotic/enhancing components to be kept.
MAX_DISTANCE_FROM_EDEMA_MM = 60.0

model_cache: dict = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NeuroVista] Using device: {device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[NeuroVista] Loading segmentation model ...")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
    model_cache["model"] = get_model(model_path, device)
    print("[NeuroVista] Model loaded successfully.")
    yield
    model_cache.clear()

app = FastAPI(title="Nextar NeuroVista AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:5500","http://127.0.0.1:5501","http://localhost:5500"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ── Spatial coherence filter ───────────────────────────────────────────────
def _keep_spatially_coherent_components(seg_mask, voxel_spacing,
                                        anchor_label=2, target_labels=(1, 3),
                                        max_distance_mm=MAX_DISTANCE_FROM_EDEMA_MM):
    """
    Remove any connected component of target_labels whose centroid is
    further than max_distance_mm from the anchor_label centroid (edema).

    WHY: After the largest-component filter, the selected "largest" enhancing
    blob may still be a large false-positive in the wrong brain region.
    Anchoring to edema guarantees all sub-regions belong to the same lesion.
    """
    sx, sy, sz = voxel_spacing
    anchor_mask = (seg_mask == anchor_label)
    if not np.any(anchor_mask):
        return seg_mask   # no anchor — skip

    av = np.argwhere(anchor_mask)
    anchor_mm = np.array([av[:,0].mean()*sx, av[:,1].mean()*sy, av[:,2].mean()*sz])

    for lbl in target_labels:
        binary = (seg_mask == lbl).astype(np.uint8)
        if not np.any(binary): continue
        labelled, num_features = ndimage_label(binary)
        kept = removed = 0
        for cid in range(1, num_features + 1):
            comp = (labelled == cid)
            vox  = np.argwhere(comp)
            centroid_mm = np.array([vox[:,0].mean()*sx, vox[:,1].mean()*sy, vox[:,2].mean()*sz])
            dist = float(np.linalg.norm(centroid_mm - anchor_mm))
            if dist <= max_distance_mm:
                kept += int(comp.sum())
            else:
                seg_mask[comp] = 0
                removed += int(comp.sum())
        label_name = {1:"Necrotic Core",2:"Edema",3:"Enhancing"}[lbl]
        print(f"[NeuroVista]   Spatial filter ({label_name}): kept {kept} vx, removed {removed} vx outside {max_distance_mm:.0f} mm")
    return seg_mask


# ── Mesh helpers ────────────────────────────────────────────────────────────
def _apply_pbr(mesh, rgba):
    r,g,b,a = [c/255.0 for c in rgba]
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[r,g,b,a], metallicFactor=0.0, roughnessFactor=0.5,
        alphaMode='BLEND' if a<1.0 else 'OPAQUE', doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    return mesh

def _build_mesh(binary_mask, spacing, face_colour, sigma=0.8, max_faces=8000):
    if not np.any(binary_mask): return None
    try:
        fm = gaussian_filter(binary_mask.astype(np.float32), sigma=sigma) if sigma>0 else binary_mask.astype(np.float32)
        verts, faces = mcubes.marching_cubes(fm, 0.5)
        verts[:,0]*=spacing[0]; verts[:,1]*=spacing[1]; verts[:,2]*=spacing[2]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if len(mesh.faces) > max_faces:
            mesh = mesh.simplify_quadric_decimation(max_faces)
        r,g,b,a = [c/255.0 for c in face_colour]
        pbr = trimesh.visual.material.PBRMaterial(
            name=f"mat_{id(binary_mask)}", baseColorFactor=[r,g,b,a],
            metallicFactor=0.0, roughnessFactor=0.7,
            alphaMode='BLEND' if a<1.0 else 'OPAQUE', doubleSided=True)
        mesh.visual = trimesh.visual.TextureVisuals(material=pbr)
        return mesh
    except Exception as exc:
        print(f"[NeuroVista] Mesh build failed — {exc}"); return None

def _line_mesh(start, end, radius=0.4, colour=None):
    try:
        c = trimesh.creation.cylinder(radius=radius, segment=[list(start),list(end)], sections=8)
        return _apply_pbr(c, colour) if colour else c
    except: return None

def _wireframe_box(bmin, bmax, radius=0.3, colour=None):
    colour = colour or [255,255,255,100]
    x0,y0,z0=bmin; x1,y1,z1=bmax
    corners=[[x0,y0,z0],[x1,y0,z0],[x0,y1,z0],[x1,y1,z0],[x0,y0,z1],[x1,y0,z1],[x0,y1,z1],[x1,y1,z1]]
    edges=[(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    cyls=[c for c in [_line_mesh(corners[i],corners[j],radius) for i,j in edges] if c]
    return _apply_pbr(trimesh.util.concatenate(cyls), colour) if cyls else None


# ── Main endpoint ────────────────────────────────────────────────────────────
@app.post("/segment-and-analyze/")
async def segment_and_analyze(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Ingest ZIP
        zip_path = os.path.join(temp_dir, file.filename)
        dicom_base_dir = os.path.join(temp_dir, "dicom_files")
        os.makedirs(dicom_base_dir)
        with open(zip_path, "wb") as buf: shutil.copyfileobj(file.file, buf)
        with zipfile.ZipFile(zip_path, "r") as zf: zf.extractall(dicom_base_dir)
        print(f"[NeuroVista] Extracted {file.filename}.")

        # 2. Resolve modalities
        modality_map = {"FLAIR":"flair","T1w":"t1","T1wCE":"t1ce","T2w":"t2"}
        nifti_paths: dict = {}
        voxel_volume_mm3 = None
        voxel_spacing = (1.0, 1.0, 1.0)

        # Path A: raw NIfTI
        for root, _, files in os.walk(dicom_base_dir):
            for fname in files:
                lf = fname.lower()
                if not (lf.endswith('.nii') or lf.endswith('.nii.gz')): continue
                fp = os.path.join(root, fname)
                if   'flair' in lf: nifti_paths['flair'] = fp
                elif 't1ce'  in lf: nifti_paths['t1ce']  = fp
                elif 't1'    in lf: nifti_paths['t1']    = fp
                elif 't2'    in lf: nifti_paths['t2']    = fp
                if voxel_volume_mm3 is None:
                    r = sitk.ImageFileReader(); r.SetFileName(fp); r.ReadImageInformation()
                    sx,sy,sz = r.GetSpacing(); voxel_volume_mm3=sx*sy*sz; voxel_spacing=(sx,sy,sz)

        # Path B: DICOM
        if len(nifti_paths) < 4:
            nifti_paths = {}
            BRATS_MAX = 4000.0
            for folder_name, mkey in modality_map.items():
                mp = os.path.join(dicom_base_dir, folder_name)
                if not os.path.isdir(mp): continue
                reader = sitk.ImageSeriesReader()
                dnames = reader.GetGDCMSeriesFileNames(mp)
                if not dnames: continue
                reader.SetFileNames(dnames); img = reader.Execute()
                if voxel_volume_mm3 is None:
                    sx,sy,sz=img.GetSpacing(); voxel_volume_mm3=sx*sy*sz; voxel_spacing=(sx,sy,sz)
                arr=sitk.GetArrayFromImage(img).astype(np.float64); amax=arr.max()
                if amax > 0 and abs(amax-BRATS_MAX) > 200:
                    arr=(arr*(BRATS_MAX/amax)).astype(np.float32)
                    rsc=sitk.GetImageFromArray(arr); rsc.CopyInformation(img); img=rsc
                    print(f"[NeuroVista]   {mkey}: DICOM rescaled [{amax:.0f}] -> [{BRATS_MAX:.0f}]")
                np_path = os.path.join(temp_dir, f"{mkey}.nii.gz")
                sitk.WriteImage(img, np_path); nifti_paths[mkey]=np_path
            print(f"[NeuroVista] Converted DICOM -> NIfTI: {list(nifti_paths.keys())}")
        else:
            print(f"[NeuroVista] Found raw NIfTI files: {list(nifti_paths.keys())}")

        # 3. Pre-processing
        required = ["flair","t1","t1ce","t2"]
        missing = [m for m in required if m not in nifti_paths]
        if missing: raise ValueError(f"Missing modalities: {missing}")

        input_dict  = {f"image_{m}": nifti_paths[m] for m in required}
        image_keys  = list(input_dict.keys())
        SPATIAL_SIZE = (240, 240, 155)

        transforms = Compose([
            LoadImaged(keys=image_keys), EnsureChannelFirstd(keys=image_keys),
            Orientationd(keys=image_keys, axcodes="RAS"),
            Spacingd(keys=image_keys, pixdim=(1.0,1.0,1.0), mode="bilinear"),
            ResizeWithPadOrCropd(keys=image_keys, spatial_size=SPATIAL_SIZE, method="symmetric", mode="constant"),
            ScaleIntensityRanged(keys=image_keys, a_min=0, a_max=4000, b_min=0.0, b_max=1.0, clip=True),
            ConcatItemsd(keys=image_keys, name="image", dim=0), ToTensord(keys=["image"]),
        ])
        processed    = transforms(input_dict)
        input_tensor = processed["image"].to(device).unsqueeze(0)
        print(f"[NeuroVista] Pre-processed tensor shape: {input_tensor.shape}")
        for ci, cn in enumerate(["FLAIR","T1","T1CE","T2"]):
            ch=input_tensor[0,ci]; print(f"[NeuroVista]   {cn}: min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}, nonzero%={100*(ch>0).float().mean():.1f}%")

        # 4. Inference
        model = model_cache["model"]
        t0 = time.perf_counter()
        with torch.no_grad():
            raw_output = sliding_window_inference(input_tensor, (96,96,96), 4, model)
        inference_time = round(time.perf_counter()-t0, 3)
        print(f"[NeuroVista] Inference done in {inference_time}s.")

        # 4a. Confidence thresholding
        sp = torch.softmax(raw_output, dim=1)
        max_probs, seg_classes = torch.max(sp, dim=1)
        max_probs = max_probs.squeeze(0).cpu().numpy()
        seg_mask  = seg_classes.squeeze(0).cpu().numpy()
        total_filtered = 0
        LNAMES = {1:"Necrotic",2:"Edema",3:"Enhancing"}
        for lid, thresh in CONFIDENCE_THRESHOLDS.items():
            unc = (seg_mask==lid)&(max_probs<thresh); n=int(unc.sum())
            if n: seg_mask[unc]=0; total_filtered+=n; print(f"[NeuroVista]   Confidence filter ({LNAMES[lid]}): removed {n} voxels (threshold={thresh})")
        if total_filtered: print(f"[NeuroVista] Total confidence-filtered: {total_filtered} voxels")

        # 4b. Clean EDEMA to its single largest component first.
        #     Edema becomes the spatial anchor — it must be clean before anything else runs.
        for lbl in [2]:
            binary=(seg_mask==lbl).astype(np.uint8)
            if not np.any(binary): continue
            labelled,nf=ndimage_label(binary)
            if nf<=1: continue
            sizes=np.array([np.sum(labelled==i) for i in range(1,nf+1)])
            lid_max=int(np.argmax(sizes))+1
            rm=(seg_mask==lbl)&(labelled!=lid_max); nr=int(rm.sum()); seg_mask[rm]=0
            print(f"[NeuroVista]   {LNAMES[lbl]} (label {lbl}): kept largest ({int(sizes[lid_max-1])} vx), removed {nr} from {nf-1} smaller blobs")

        # 4c. Spatial coherence BEFORE largest-component for necrotic & enhancing.
        #
        #   PREVIOUS (broken) order:
        #     largest-component → always picked the ectopic cluster (biggest blob)
        #     → spatial filter removed everything → enhancing = 0
        #
        #   CORRECT order (now):
        #     spatial filter first → eliminates all blobs far from edema centroid
        #     → largest-component → picks the biggest blob that is near the real tumour
        #
        seg_mask = _keep_spatially_coherent_components(seg_mask, voxel_spacing,
                                                        anchor_label=2, target_labels=(1,3),
                                                        max_distance_mm=MAX_DISTANCE_FROM_EDEMA_MM)

        # 4d. NOW run largest-component on the spatially-filtered survivors.
        for lbl in [1, 3]:
            binary=(seg_mask==lbl).astype(np.uint8)
            if not np.any(binary): continue
            labelled,nf=ndimage_label(binary)
            if nf<=1: continue
            sizes=np.array([np.sum(labelled==i) for i in range(1,nf+1)])
            lid_max=int(np.argmax(sizes))+1
            rm=(seg_mask==lbl)&(labelled!=lid_max); nr=int(rm.sum()); seg_mask[rm]=0
            print(f"[NeuroVista]   {LNAMES[lbl]} (label {lbl}): kept largest near-edema blob ({int(sizes[lid_max-1])} vx), removed {nr} noise blobs")

        tumor_pct = 100*np.sum(seg_mask>0)/seg_mask.size
        print(f"[NeuroVista] Post-processed tumor fraction: {tumor_pct:.2f}% of volume")

        # Log final centroids
        for lbl, name in {1:"Necrotic Core",2:"Edema",3:"Enhancing"}.items():
            vox=np.argwhere(seg_mask==lbl)
            if len(vox)>0:
                cx,cy,cz=vox[:,0].mean()*voxel_spacing[0],vox[:,1].mean()*voxel_spacing[1],vox[:,2].mean()*voxel_spacing[2]
                print(f"[NeuroVista]   {name} centroid (mm): ({cx:.1f}, {cy:.1f}, {cz:.1f})")

        brain_mask=(input_tensor[0,1].cpu().numpy()>0.01).astype(np.uint8)

        # 5. Volume calculation
        volume_report = calculate_clinical_volumes(seg_mask, voxel_volume_mm3)
        volume_report["metadata"] = {"inference_time_seconds": inference_time, "warnings": volume_report.pop("warnings",[])}
        print(f"[NeuroVista] Volumes — Total: {volume_report['volumes_mm3']['total_tumor']:.1f} mm3  |  Edema: {volume_report['volumes_mm3']['edema']:.1f} mm3  |  Enhancing: {volume_report['volumes_mm3']['enhancing_tumor']:.1f} mm3")

        # 6. Meshes
        print("[NeuroVista] Building smoothed 3-D meshes ...")
        brain_mesh     = _build_mesh(brain_mask,    voxel_spacing, COLOUR_BRAIN,     sigma=SIGMA_BRAIN,     max_faces=12000)
        edema_mesh     = _build_mesh(seg_mask==2,   voxel_spacing, COLOUR_EDEMA,     sigma=SIGMA_EDEMA,     max_faces=6000)
        necrotic_mesh  = _build_mesh(seg_mask==1,   voxel_spacing, COLOUR_NECROTIC,  sigma=SIGMA_NECROTIC,  max_faces=4000)
        enhancing_mesh = _build_mesh(seg_mask==3,   voxel_spacing, COLOUR_ENHANCING, sigma=SIGMA_ENHANCING, max_faces=4000)

        # 6b. Surgical planning
        print("[NeuroVista] Computing surgical planning geometry ...")
        surgical_planning = {}; surgical_meshes = []
        tumor_parts=[m for m in [edema_mesh,necrotic_mesh,enhancing_mesh] if m is not None]
        if tumor_parts and brain_mesh is not None:
            ct=trimesh.util.concatenate(tumor_parts); tc=ct.centroid
            surgical_planning["tumor_centroid"]=tc.tolist()
            bmin,bmax=ct.bounds; bd=bmax-bmin
            surgical_planning.update({"bbox_min":bmin.tolist(),"bbox_max":bmax.tolist(),"bbox_dims_mm":[round(d,1) for d in bd.tolist()]})
            cpts,dists,_=trimesh.proximity.closest_point(brain_mesh,[tc])
            spt=cpts[0]; cdepth=float(dists[0])
            surgical_planning.update({"brain_surface_point":spt.tolist(),"cortical_depth_mm":round(cdepth,1)})
            bcx=(brain_mesh.bounds[0][0]+brain_mesh.bounds[1][0])/2.0
            tx=ct.vertices[:,0]; midx=np.argmin(tx) if tc[0]>bcx else np.argmax(tx)
            mpt=ct.vertices[midx].copy(); mdist=float(abs(mpt[0]-bcx)); mwarn=mdist<5.0
            mtgt=mpt.copy(); mtgt[0]=bcx
            surgical_planning.update({"midline_x":round(float(bcx),1),"midline_distance_mm":round(mdist,1),"midline_tumor_point":mpt.tolist(),"midline_target_point":mtgt.tolist(),"midline_warning":mwarn})
            print(f"[NeuroVista]   Cortical depth: {cdepth:.1f} mm  |  Midline: {mdist:.1f} mm{'  WARNING' if mwarn else ''}")
            sph=trimesh.creation.icosphere(radius=3.5,subdivisions=2); sph.apply_translation(tc); _apply_pbr(sph,[0,220,255,255]); surgical_meshes.append(("surg_centroid",sph))
            wf=_wireframe_box(bmin,bmax,0.8,[255,255,255,120]);
            if wf: surgical_meshes.append(("surg_bbox",wf))
            corr=_line_mesh(spt,tc,1.5,[0,220,255,220]);
            if corr: surgical_meshes.append(("surg_corridor",corr))
            ml=_line_mesh(mpt,mtgt,1.2,[255,40,40,255] if mwarn else [180,255,0,220]);
            if ml: surgical_meshes.append(("surg_midline",ml))

        # 7. GLB scene
        scene=trimesh.Scene(); mesh_centroids={}
        for name,mesh in [("brain",brain_mesh),("edema",edema_mesh),("necrotic_core",necrotic_mesh),("enhancing_tumor",enhancing_mesh)]:
            if mesh is not None:
                scene.add_geometry(mesh,node_name=name)
                c=mesh.vertices.mean(axis=0).tolist(); mesh_centroids[name]=c
                print(f"[NeuroVista]   + {name}: {len(mesh.faces)} faces  centroid=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")
        for name,mesh in surgical_meshes:
            scene.add_geometry(mesh,node_name=name); print(f"[NeuroVista]   + {name}: {len(mesh.faces)} faces")

        glb_bytes=trimesh.exchange.gltf.export_glb(scene)
        debug_dir=os.path.join(os.path.dirname(__file__),"..","debug_output"); os.makedirs(debug_dir,exist_ok=True)
        safe_stem=os.path.splitext(os.path.basename(file.filename or "scene"))[0]
        out_path=os.path.join(debug_dir,f"{safe_stem}_{int(time.time())}.glb")
        with open(out_path,"wb") as f: f.write(glb_bytes)
        print(f"[NeuroVista] Debug GLB saved -> {out_path}")
        glb_base64=base64.b64encode(glb_bytes).decode("utf-8")
        print("[NeuroVista] Scene encoded.")

        # 8. Response
        cdss=generate_clinical_report(volume_report)
        clinical_report={
            "volumes_mm3":volume_report["volumes_mm3"],"derived_metrics":volume_report["derived_metrics"],
            "metadata":volume_report["metadata"],"triage_band":cdss["triage_band"],
            "malignancy_likelihood_pct":cdss["malignancy_likelihood_pct"],"grade_hint":cdss["grade_hint"],
            "grade_probabilities":cdss["grade_probabilities"],"outcome_range":cdss["outcome_range"],
            "surgical_note":cdss["surgical_note"],"disclaimer":cdss["disclaimer"],
        }
        return JSONResponse(content={"clinical_report":clinical_report,"model_glb_base64":glb_base64,"mesh_centroids":mesh_centroids,"surgical_planning":surgical_planning,"message":"Segmentation successful."})

    except Exception as exc:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        print(f"[NeuroVista] Cleaning up: {temp_dir}"); shutil.rmtree(temp_dir,ignore_errors=True)

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="127.0.0.1", port=8000)