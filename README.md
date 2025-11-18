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