# NeuroVista AI: Complete Project Deep Dive

This document serves as a comprehensive explanation of every component, feature, and architectural decision in the NeuroVista AI 3D Brain Tumor Segmentation project. It covers both the frontend user interface and the intricate deep learning and processing pipelines on the backend.

---

## 1. System Architecture Overview

The system is a full-stack web application designed for medical image analysis. It takes raw MRI scans of a patient's brain, uses a deep learning model to find the tumor and its sub-regions, and presents a clinical report alongside an interactive 3D model.

- **Frontend:** HTML/CSS/JS (Vanilla) using modern web standards, glassmorphism UI, and Google's `<model-viewer>` for 3D rendering.
- **Backend:** Python using **FastAPI** for high-performance API routing.
- **Deep Learning Engine:** PyTorch and **MONAI** (Medical Open Network for AI).
- **Medical Imaging Stack:** SimpleITK (for DICOM/NIfTI handling), scikit-image (for mesh generation), and Trimesh (for 3D geometry manipulation).

---

## 2. Frontend Interface (`index.html`)

The frontend is a single-page application built to look like a professional, clinical-grade medical dashboard.

### 2.1 UI/UX Design & Theming
- **Modern Aesthetic:** Built with a customized CSS variable system. It uses the "Inter" font for clean readability, glassmorphism (blurred, semi-transparent backgrounds), and smooth hover micro-animations.
- **Dark/Light Mode:** Full support for theme toggling. Variables like `--bg-color`, `--box-bg-color`, and `--text-color` swap seamlessly to provide a comfortable viewing experience in different lighting environments.

### 2.2 Interactive 3D Visualization (`<model-viewer>`)
The core of the frontend is the 3D viewer. Once the backend processes the scan, it returns a Base64 encoded `.glb` 3D model. 
- **Legend & Controls:** The user can freely rotate and zoom the 3D brain. A legend indicates the colors: Brain shell (transparent blue/grey), Edema (yellow), Necrotic core (dark red), and Enhancing tumor (bright red).

### 2.3 Interactive 3D Tooltips (Hotspots)
To make the 3D model clinically useful, the UI dynamically injects **HTML hotspots** into the `<model-viewer>` environment based on the 3D coordinates (centroids) provided by the backend.
- **Pulsing Animation:** Each sub-region has a pulsing dot.
- **Hover Emphasis:** Hovering over a dot highlights that specific tumor region and dims the rest.
- **Data Display:** A floating glassmorphism tooltip card appears, showing the region's clinical name, its exact volume in mm³, its percentage of the total lesion, and a medical explanation of what the region represents.

### 2.4 AI Surgical Planning Assessment
A dedicated toggle reveals the **Surgical Planning Panel**. This is powered by spatial calculations from the backend.
- **Metrics Displayed:** Shows the Minimal Cortical Depth (distance from the tumor to the brain surface), Midline Proximity (distance to the brain's center dividing line), and physical bounding box dimensions (Width, Height, Depth).
- **Dynamic Risk Warnings:** If the tumor is within 5mm of the mid-sagittal plane, the UI highlights the metric in red as a "High Risk" warning.
- **3D Integration:** Toggling this panel also reveals surgical wireframes inside the 3D viewer (e.g., a line drawing the shortest path to the surface).

---

## 3. Backend API & Pipeline (`app/main.py`)

The FastAPI backend is the workhorse of the application. The main endpoint (`POST /segment-and-analyze/`) orchestrates a complex, multi-stage pipeline.

### 3.1 Data Ingestion & Conversion
1. **ZIP Extraction:** The API receives a `.zip` file containing a patient's MRI scans and extracts it to a temporary directory.
2. **DICOM to NIfTI Conversion:** Raw medical scans usually come as "DICOM" files (hundreds of 2D slice images). The backend uses `SimpleITK` to intelligently group these slices and convert them into volumetric 3D "NIfTI" (`.nii.gz`) files.
3. **Intensity Normalization:** It detects the 4 required MRI modalities (FLAIR, T1, T1ce, T2). If the pixel intensities are vastly out of range, it dynamically rescales them to match the standard BraTS dataset range (max 4000).

### 3.2 Preprocessing (MONAI)
Before the AI can look at the images, they must be standardized using MONAI transforms:
- `Orientationd`: Rotates the brain into a standard RAS (Right, Anterior, Superior) anatomical orientation.
- `Spacingd`: Forces the voxels (3D pixels) to be exactly 1mm x 1mm x 1mm. This is crucial so physical volume calculations are accurate.
- `ResizeWithPadOrCropd`: Forces the volume into a standard `240x240x155` shape.
- `ScaleIntensityRanged`: Normalizes the signal intensity to a float between 0.0 and 1.0.

### 3.3 Deep Learning Inference
- **3D U-Net:** The pre-processed 4-channel tensor is fed into the loaded PyTorch 3D U-Net model.
- **Sliding Window:** Because the brain is too large to fit into GPU memory all at once, `sliding_window_inference` runs the model on small `96x96x96` chunks and stitches the predictions together.

### 3.4 Advanced Post-Processing (Artifact Removal)
Deep learning models often "hallucinate" false positive tumors in healthy parts of the brain. The backend features a robust, multi-step filter to guarantee anatomical accuracy:
1. **Confidence Thresholding:** Voxels where the model's confidence is too low (e.g., < 70% for enhancing core) are deleted.
2. **Edema Anchoring:** The script isolates the "Edema" (swelling) prediction and deletes everything except the single largest connected blob. This becomes the "Anchor".
3. **Spatial Coherence Filter:** A custom algorithm calculates the physical Z, Y, X centroid of the Edema anchor. Any piece of predicted "Necrotic" or "Enhancing" tumor that is further than `60mm` away from the edema is immediately deleted. This completely prevents the model from hallucinating a second tumor in the wrong hemisphere.
4. **Largest Component (Core):** Finally, it cleans up the remaining core regions, keeping only the largest valid blob for each.

### 3.5 3D Mesh Generation & Surgical Geometry
- **Marching Cubes:** The 3D voxel mask is passed through a Gaussian filter (for smoothness) and then through the `marching_cubes` algorithm to generate 3D geometry (vertices and faces).
- **Decimation:** Medical meshes are incredibly dense. `trimesh.simplify_quadric_decimation` dramatically reduces the polygon count (e.g., from 100,000 faces down to 4,000 faces) so the web browser can render it smoothly without lagging.
- **Surgical Calculations:** 
  - `trimesh.proximity.closest_point` calculates the shortest physical distance from the tumor centroid to the outer brain shell (Cortical Depth).
  - The midline proximity is calculated by measuring the X-axis distance from the tumor's medial edge to the brain's mathematical center.
- **Export:** All meshes (brain, tumor parts, and surgical wires) are packed into a single `.glb` scene, colored using PBR materials (transparency, roughness), and encoded to Base64 to be sent via JSON.

---

## 4. Clinical Logic Modules

To make the application a true "Clinical Decision Support System" (CDSS), raw voxel counts are transformed into actionable medical intelligence.

### 4.1 Volumetric Calculation (`app/processing.py`)
- The `calculate_clinical_volumes` function counts the number of voxels for each label and multiplies it by the physical voxel volume (`1 mm³`).
- It calculates standard neuro-oncology metrics: **Total Tumor Volume**, **Total Core Volume**, and the **Enhancing Fraction** (what percentage of the core is actively taking up contrast).
- It generates QA Warnings (e.g., flagging the radiologist if the enhancing fraction is impossibly high, or if the tumor volume is suspiciously small).

### 4.2 Clinical Decision Support (`app/clinical_decision.py`)
This module takes the volumes and acts like an AI neuro-oncologist:
- **Malignancy Likelihood Score (0-100%):** Uses a weighted formula primarily driven by the enhancing fraction and the absolute volume of the core.
- **Grade Classification:** Predicts the probabilities of the tumor being WHO Grade I-II (low grade), III, or IV (Glioblastoma) based on size and necrosis/enhancement patterns.
- **Triage Banding:** Automatically flags cases as **URGENT** (requires same-day review), **EXPEDITED**, or **ROUTINE** based on aggressive features (e.g., a massive enhancing core or midline shifting risk).
- **Outcome Range & Surgical Note:** Generates literature-grounded text snippets advising on expected survival ranges and surgical recommendations (like whether an awake craniotomy should be considered based on proximity to eloquent areas).

---

## Summary of the Workflow

1. User uploads ZIP file of MRI scans to the beautiful web UI.
2. Backend API extracts, normalizes, and aligns the scans into a 3D tensor.
3. The 3D U-Net model predicts the location of the tumor.
4. The Post-Processing pipeline destroys false positives using spatial coherence rules.
5. The system calculates exact volumes and runs them through a clinical logic engine to predict urgency and tumor grade.
6. The voxels are smoothed and converted into an optimized 3D GLB mesh.
7. The UI receives the data, rendering the 3D model, the interactive clinical hotspots, and the AI Surgical Assessment dashboard.

---

## 5. Methodology

The proposed methodology implements an end-to-end pipeline for automated brain tumor analysis, combining multimodal MRI preprocessing, deep learning-based segmentation, quantitative volumetry, and interactive 3D reconstruction. The workflow is designed to bridge the gap between voxel-level prediction and clinically interpretable visualization.

### 5.1 Dataset Acquisition

The system is developed and evaluated using the BraTS 2020 benchmark dataset, which contains pre-operative, multi-institutional brain MRI studies of glioma patients. Each subject includes four co-registered MRI modalities: FLAIR, T1-weighted (T1), contrast-enhanced T1-weighted (T1ce), and T2-weighted (T2). These modalities provide complementary anatomical and pathological information, including edema, structural anatomy, contrast enhancement, and fluid-sensitive tissue contrast.

The reference segmentation labels are harmonized to a continuous label space for model training and inference. In the implementation, the original BraTS label 4 is remapped to label 3 so that the output classes become background (0), necrotic/non-enhancing core (1), edema (2), and enhancing tumor (3).

### 5.2 Data Preprocessing

To reduce inter-patient variability and scanner-related heterogeneity, all inputs are standardized using MONAI-based preprocessing. The pipeline applies the following operations:

- Spatial orientation standardization using the RAS anatomical convention.
- Isotropic resampling with 1 mm x 1 mm x 1 mm voxel spacing.
- Intensity normalization to a fixed range to improve numerical stability during training and inference.
- Padding and cropping to a fixed spatial size of 240 x 240 x 155 voxels.
- Channel concatenation to form a 4-channel volumetric input tensor.

During inference, the backend accepts either raw NIfTI volumes or zipped DICOM series. When DICOM data are provided, SimpleITK is used to reconstruct 3D volumes from the slice stacks and preserve spatial metadata required for correct resampling and physical volume estimation.

### 5.3 Model Architecture

The segmentation model is a 3D U-Net implemented in PyTorch with MONAI. The architecture follows an encoder-decoder design:

- The encoder progressively reduces spatial resolution while learning abstract contextual features.
- The decoder restores spatial resolution through upsampling operations.
- Skip connections transfer fine-grained localization information from encoder layers to the decoder, improving boundary accuracy.

The network takes a 4-channel 3D input corresponding to the four MRI modalities and produces a voxel-wise multi-class probability map over the four output classes.

### 5.4 Training Protocol

Model training is performed in a GPU-accelerated environment using a patch-based strategy to manage the memory demands of 3D volumes. Training patches are sampled from the standardized MRI volumes, and balanced sampling is used so that tumor-containing patches are sufficiently represented despite the class imbalance typical of medical segmentation problems.

The optimization setup includes:

- Dice + Cross-Entropy loss to jointly optimize overlap quality and voxel-wise classification accuracy.
- Adam optimization with a learning rate of 1 x 10^-4.
- ReduceLROnPlateau scheduling to lower the learning rate when validation performance saturates.
- Early stopping to prevent overfitting and retain the best model according to validation Dice score.

### 5.5 Inference and 3D Reconstruction

For deployment, the trained model is exposed through a FastAPI service. The endpoint accepts a zipped study, extracts the modality volumes, applies the same preprocessing pipeline used during training, and runs 3D inference using a sliding-window strategy with overlap to accommodate full-size brain volumes that exceed the memory limits of a single forward pass.

After inference, the predicted segmentation mask undergoes post-processing to improve anatomical consistency. Low-confidence voxels are removed, connected components are filtered to retain the most plausible lesion regions, and edema is used as an anatomical anchor to suppress spatially implausible tumor fragments.

The final segmentation mask is then used for:

- Volumetric quantification of edema, necrotic core, enhancing tumor, and total lesion volume.
- Mesh generation using the Marching Cubes algorithm.
- Mesh simplification with Trimesh so that the 3D output can be rendered efficiently in a web browser.
- Export of the final visualization as a GLB scene for interactive viewing in the frontend.

### 5.6 Clinical Interpretation Layer

In addition to segmentation, the pipeline computes derived clinical indicators from the predicted volumes. These include total tumor volume, total core volume, enhancing fraction, malignancy likelihood, grade-related heuristics, and triage categories. This rule-based post-processing layer is intended to support clinical interpretation by converting segmented output into actionable summary metrics, while still requiring expert review before any medical decision is made.

### 5.7 Summary of Methodological Contribution

The main methodological contribution of this work is the integration of multimodal 3D segmentation, volumetric analysis, spatially coherent post-processing, and interactive 3D visualization into a single deployable pipeline. By combining deep learning with geometrical reconstruction and clinical summarization, the system produces outputs that are both computationally meaningful and easier to interpret in a medical workflow.
