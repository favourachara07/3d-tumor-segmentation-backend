# 3D Brain Tumor Segmentation API

This project is the backend service for a 3D medical imaging analysis application. It uses a trained 3D U-Net model (built with MONAI and PyTorch) to perform semantic segmentation on multi-modal brain MRI scans.

The API is built with FastAPI and is designed to:
1.  Accept a ZIP file containing DICOM series (FLAIR, T1, T1ce, T2).
2.  Preprocess the images and run inference with a 3D U-Net.
3.  Calculate the tumor volume in cubic millimeters.
4.  Generate a simplified 3D mesh (.glb format) of the segmented tumor.
5.  Return the volume and the mesh to a frontend client.

## Setup and Installation

### 1. Model Placement
This repository does not include the trained model file due to its size.
- **Download the model:** You can download the trained model (`best_model.pth`) from [LINK TO YOUR KAGGLE NOTEBOOK OUTPUT OR GOOGLE DRIVE].
- **Place the model:** Create a `models/` directory in the project root and place the `best_model.pth` file inside it.

### 2. Environment Setup
It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\\Scripts\\activate
# Or (macOS/Linux)
# source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

# Deep Learning Core: 3D Brain Tumor Segmentation

This document details the machine learning pipeline used to build the 3D segmentation model for this project. As my first deep learning project, this work represents a practical application of Convolutional Neural Networks (CNNs) to complex, volumetric medical data.

## 1. Conceptual Foundation: From 2D CNNs to 3D Segmentation

### Understanding the Basics
A traditional **Convolutional Neural Network (CNN)** operates on 2D images (like a photo of a cat). It scans the image with small filters (kernels) to detect features like edges, textures, and eventually, complex shapes.

### The Challenge of 3D Medical Data
An MRI scan is not a flat image; it is a 3D volume composed of stacked 2D slices (voxels instead of pixels). A simple 2D CNN would lose critical spatial context along the depth (Z) axis. For example, a tumor usually persists across multiple slices. To capture this, we need a model that can "see" in 3D.

### The Solution: 3D U-Net
For this project, I utilized the **3D U-Net architecture**, the industry standard for medical image segmentation.
*   **Encoder (Contracting Path):** The model applies 3D convolutions to capture context (what the tumor looks like), reducing the image size while increasing feature complexity.
*   **Decoder (Expanding Path):** The model upsamples the features back to the original size to enable precise localization (where the tumor is).
*   **Skip Connections:** Critical links between the encoder and decoder layers that preserve fine-grained details lost during downsampling, ensuring the segmentation mask has sharp, accurate boundaries.

---

## 2. The Tech Stack

The project leverages a specialized stack of libraries designed for deep learning and medical imaging.

### Core Framework
*   **PyTorch (`torch`):** The underlying deep learning framework. It provides the tensor data structures and automatic differentiation needed to train neural networks.
    *   *Why `cu121`?* The specific installation command (`--index-url .../cu121`) ensures we install the version of PyTorch optimized for CUDA 12.1, allowing the model to train on NVIDIA GPUs for massive speedups.

### The Medical AI Specialist
*   **MONAI (`monai[all]`):** The **Medical Open Network for AI**. This library sits on top of PyTorch and abstracts away the massive complexity of handling medical data.
    *   **Role:** Instead of writing custom code to read NIfTI files, handle 3D data augmentation, or build a 3D U-Net from scratch, MONAI provides optimized, pre-built components for all these tasks. It is the engine that makes this project possible.

### Data Processing & Utilities
*   **`nibabel`:** A low-level library used to read and write neuroimaging files (specifically the `.nii` / NIfTI format used in the BraTS dataset).
*   **`simpleitk` (Sitk):** A powerful image analysis toolkit.
    *   *In Backend:* Used to read directories of raw DICOM files (the standard format from hospital scanners) and robustly convert them into the NIfTI format the model expects. It handles spacing, orientation, and metadata alignment.
*   **`scikit-image`:** A collection of algorithms for image processing.
    *   *Key Function:* `marching_cubes`. This algorithm takes the raw 3D voxel cloud produced by the model (the segmentation mask) and converts it into a geometric mesh (vertices and faces) that can be rendered in 3D.
*   **`trimesh`:** A library for manipulating 3D meshes.
    *   *Role:* Used to process the raw mesh from `marching_cubes`, simplify it (reduce polygon count) for web performance, and export it as a `.glb` file for the frontend.
*   **`tqdm`:** Provides progress bars (e.g., during training loops), essential for monitoring long-running processes.

---

## 3. Model Training (The Notebook)

The training process was conducted in a Kaggle notebook to leverage free GPU resources.

### A. Data Preparation
The model was trained on the **BraTS 2020 dataset**, a benchmark dataset for brain tumor segmentation.
1.  **Multi-Modal Input:** Each patient sample consists of **4 different MRI scans** (modalities):
    *   **FLAIR:** Fluid-Attenuated Inversion Recovery (good for edema).
    *   **T1:** T1-weighted (structural).
    *   **T1ce:** T1-weighted contrast-enhanced (good for active tumor).
    *   **T2:** T2-weighted.
2.  **The Input Tensor:** These 4 scans are stacked into a single 4-channel 3D tensor `(4, H, W, D)`.

### B. Preprocessing Pipeline (MONAI Transforms)
Raw medical data is messy. A robust pipeline was built using MONAI transforms:
*   **`LoadImaged`:** Reads the 4 files into memory.
*   **`EnsureChannelFirstd`:** Reformats data to PyTorch standards `(Channels, Spatial...)`.
*   **`Orientationd`:** Re-orients all brains to a standard position (RAS - Right, Anterior, Superior) so the model doesn't have to learn rotation.
*   **`Spacingd`:** **Crucial Step.** Resamples all images to a standard resolution (1mm x 1mm x 1mm voxels). This ensures physical size is consistent across patients, regardless of the scanner used.
*   **`ScaleIntensityRanged`:** Normalizes pixel values to a 0-1 range for stable training.
*   **`CropForegroundd`:** Crops out the empty black space around the head to save memory.
*   **`RandCropByPosNegLabeld`:** Extracts smaller 3D patches (e.g., `96x96x96`) from the full brain for training. It ensures a balance of healthy tissue and tumor tissue in every batch.

### C. Training Loop
*   **Loss Function:** `DiceCELoss`. A combination of Dice Loss (measures overlap quality) and Cross-Entropy Loss. This combination is robust against the **class imbalance** problem (tumors are very small compared to the rest of the brain).
*   **Optimizer:** `Adam`. An adaptive learning rate optimization algorithm.
*   **Scheduler:** `ReduceLROnPlateau`. Automatically lowers the learning rate when the validation score stops improving, allowing the model to fine-tune its weights.
*   **Early Stopping:** Monitored the validation Dice score and stopped training automatically when the model stopped improving to prevent overfitting.

---

## 4. Backend Inference System

The trained model (`best_model.pth`) is deployed via a **FastAPI** backend. The inference process mirrors the training but adapts it for real-world usage.

### The Challenge: Real-World Data
In the wild, data comes as **DICOM** series (folders of .dcm slices), often with varying resolutions and field-of-views. The model expects aligned, isotropic **NIfTI** tensors.

### The Inference Pipeline
1.  **Ingestion:** The API accepts a ZIP file containing DICOM series.
2.  **Intelligent Conversion:** `SimpleITK` scans the folder, groups files by Series UID, identifies the modality (FLAIR, T1, etc.), and converts each series into a 3D NIfTI volume in memory.
3.  **Robust Preprocessing:**
    *   Each of the 4 modalities is loaded and processed **independently** using `Spacingd` and `Orientationd`. This ensures that even if the T1 scan has a different resolution than the FLAIR scan, they are both resampled to the exact same 1mm grid *before* being stacked.
    *   `ConcatItemsd` stacks them into the required 4-channel tensor.
    *   `ResizeWithPadOrCropd` forces the tensor to a standard size (e.g., `240x240x155`) to match the model's expected input geometry.
4.  **Sliding Window Inference:**
    *   Since the full brain volume is larger than the training patch size (`96x96x96`), we cannot feed it all at once.
    *   MONAI's `sliding_window_inference` moves a window across the large input volume, generating predictions for each chunk, and stitching them together into a seamless 3D segmentation map.
5.  **Post-Processing & Visualization:**
    *   **Volume Calculation:** We count the non-background voxels in the prediction mask and multiply by the voxel volume (1mm³) to get the clinical tumor volume.
    *   **3D Mesh Generation:**
        1.  `marching_cubes` (scikit-image) converts the voxel mask into a high-resolution mesh.
        2.  `trimesh.decimate` (trimesh) simplifies this mesh (e.g., reduces 100k faces to 15k faces) to make it lightweight enough to render in a web browser without crashing.
    *   **Export:** The final lightweight mesh is exported as `.glb` (Base64 encoded) and sent to the frontend.