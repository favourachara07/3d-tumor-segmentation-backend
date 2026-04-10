import torch
import numpy as np
from monai.networks.nets import UNet


def get_model(model_path: str, device: torch.device = None) -> UNet:
    """
    Loads the 3D U-Net model and its trained weights onto the specified device.
    BraTS label mapping used during training:
        0 → Background
        1 → Necrotic and Non-Enhancing Tumor Core (NCR/NET)
        2 → Peritumoral Edema (ED)
        3 → GD-Enhancing Tumor (ET)  [remapped from BraTS original label 4]
    """
    if device is None:
        device = torch.device("cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def calculate_clinical_volumes(
    seg_mask_np: np.ndarray,
    voxel_volume_mm3: float,
) -> dict:
    """
    Derive all clinically relevant volumetric metrics from a segmentation mask.

    Parameters
    ----------
    seg_mask_np : np.ndarray
        Integer array of shape (H, W, D) with values in {0, 1, 2, 3}.
    voxel_volume_mm3 : float
        Physical volume of one voxel in mm³ (product of spacing in all 3 axes).

    Returns
    -------
    dict
        Nested dict ready to be embedded in the ``clinical_report`` JSON key.

    Volume definitions
    ------------------
    - necrotic_core  → label 1 voxels × voxel_volume
    - edema          → label 2 voxels × voxel_volume
    - enhancing      → label 3 voxels × voxel_volume
    - total_tumor    → necrotic + edema + enhancing   (whole tumour region)
    - total_core     → necrotic + enhancing            (tumour core, BraTS TC metric)
    - enhancing_fraction (%) → enhancing / total_core × 100
                               Returns 0.0 if total_core == 0 to avoid ZeroDivisionError.
    """
    vol_necrotic   = float(np.sum(seg_mask_np == 1) * voxel_volume_mm3)
    vol_edema      = float(np.sum(seg_mask_np == 2) * voxel_volume_mm3)
    vol_enhancing  = float(np.sum(seg_mask_np == 3) * voxel_volume_mm3)

    total_tumor = vol_necrotic + vol_edema + vol_enhancing
    total_core  = vol_necrotic + vol_enhancing

    # Enhancing fraction — guard against empty-core edge case
    if total_core > 0.0:
        enhancing_fraction = (vol_enhancing / total_core) * 100.0
    else:
        enhancing_fraction = 0.0

    # Build warnings list for clinical QA
    warnings = []
    if total_tumor < 100.0:                         # < 100 mm³  ≈ very small / likely FP
        warnings.append("Total tumour volume is very small (< 100 mm³). Consider reviewing segmentation.")
    if enhancing_fraction > 90.0:
        warnings.append("Enhancing fraction > 90 % — unusually high. Verify model output.")
    if vol_edema == 0.0 and total_tumor > 0.0:
        warnings.append("No edema detected alongside tumour mass. May indicate limited FLAIR signal.")

    return {
        "volumes_mm3": {
            "total_tumor":      round(total_tumor,   2),
            "edema":            round(vol_edema,     2),
            "necrotic_core":    round(vol_necrotic,  2),
            "enhancing_tumor":  round(vol_enhancing, 2),
        },
        "derived_metrics": {
            "total_core":           round(total_core,         2),
            "enhancing_fraction":   round(enhancing_fraction, 4),   # percentage
        },
        "warnings": warnings,
    }