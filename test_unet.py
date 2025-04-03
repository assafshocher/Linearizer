import torch
import torch.nn as nn
from models import Unet
import argparse

# Define a simple test function

def test_unet():
    # Configuration for the Unet
    conf = argparse.Namespace(
        im_shape=(1, 64, 64),  # Example image shape (C, H, W)
        n_levels=5,  # Number of resolution levels
        base_chans=64,
    )
    
    # Initialize the Unet
    unet = Unet(conf)
    
    # Create synthetic data
    batch_size = 4
    x = torch.randn(batch_size, *conf.im_shape)  # Random input tensor
    
    # Forward pass through the UNet
    output = unet(x, debug=True)
    
    # Check output shape
    assert output.shape == x.shape, f"Output shape {output.shape} does not match input shape {x.shape}"
    print("UNet test passed: Output shape matches input shape.")

if __name__ == "__main__":
    test_unet()
