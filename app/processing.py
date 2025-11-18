import torch
from monai.networks.nets import UNet

def get_model(model_path, device):
    """Loads the UNet model and its trained weights."""
    model = UNet(
        spatial_dims=3, in_channels=4, out_channels=4,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model