import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F

from tinygrad import Tensor
from tinygrad import nn
from tinygrad.nn import InstanceNorm

def test_instancenorm(input_shape, num_features):
    # Create numpy arrays for input
    input_np = np.random.randn(*input_shape).astype('float32')
    
    # Convert numpy array to PyTorch tensor
    input_tensor_torch = torch.from_numpy(input_np)
    in_layer_torch = tnn.InstanceNorm3d(num_features, affine=True)
    in_layer_torch.weight.data = torch.from_numpy(np.load("bn_weights.npy"))
    in_layer_torch.bias.data = torch.from_numpy(np.load("bn_bias.npy"))
    output_torch = in_layer_torch(input_tensor_torch)
    
    # TinyGrad
    input_tensor_tg = Tensor(input_np)
    in_layer_tg = InstanceNorm(num_features, affine=True)
    in_layer_tg.weight = Tensor(np.load("bn_weights.npy"))
    in_layer_tg.bias = Tensor(np.load("bn_bias.npy"))
    output_tg = in_layer_tg(input_tensor_tg)

    print(f"Output shape (PyTorch): {output_torch.shape}")
    print()
    
    print(f"Input shape: {input_shape}")
    print(f"Number of features: {num_features}")
    print()

    # Check if the shapes of the outputs match
    if output_torch.shape != output_tg.shape:
        print("Output shapes do not match:")
        print(f"PyTorch output shape: {output_torch.shape}")
        print(f"TinyGrad output shape: {output_tg.shape}")
    else:
        # Convert PyTorch tensor to numpy array
        output_torch_np = output_torch.detach().numpy()

        # Convert TinyGrad tensor to numpy array
        output_tg_np = output_tg.numpy()

        # Compare the outputs
        if np.allclose(output_torch_np, output_tg_np, rtol=1e-4, atol=1e-4):
            print("Outputs match!")
        else:
            print("Outputs do not match.")

            # Calculate the absolute difference between the outputs
            diff = np.abs(output_torch_np - output_tg_np)
            print(f"Max absolute difference: {np.max(diff)}")
            print(f"Mean absolute difference: {np.mean(diff)}")

# Test the InstanceNorm layer
input_shape = (1, 21, 256, 256, 256)
num_features = 21

test_instancenorm(input_shape, num_features)
