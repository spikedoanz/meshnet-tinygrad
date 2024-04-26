import numpy as np
import torch
import torch.nn as tnn

from tinygrad import Tensor
from tinygrad import nn

def test_conv3d(input_shape, weight_shape, bias_shape=None, stride=1, padding=0, dilation=1, groups=1):
    # Create numpy arrays for input, weight, and bias
    input_np = np.random.randn(*input_shape).astype('float32')
    weight_np = np.random.randn(*weight_shape).astype('float32')
    bias_np = np.random.randn(*bias_shape).astype('float32') if bias_shape else None

    # Convert numpy arrays to PyTorch tensors
    input_tensor_torch = torch.from_numpy(input_np).float()
    weight_tensor_torch = torch.from_numpy(weight_np).float()
    bias_tensor_torch = torch.from_numpy(bias_np).float() if bias_np is not None else None

    # Perform Conv3d using PyTorch
    output_torch = tnn.functional.conv3d(input_tensor_torch, weight_tensor_torch, bias_tensor_torch, stride, padding, dilation, groups)

    # Convert numpy arrays to TinyGrad tensors
    input_tensor_tg = Tensor(input_np)
    weight_tensor_tg = Tensor(weight_np)
    bias_tensor_tg = Tensor(bias_np) if bias_np is not None else None

    # Perform Conv3d using TinyGrad
    output_tg = input_tensor_tg.conv2d(weight_tensor_tg, bias_tensor_tg, groups=groups, stride=stride, dilation=dilation, padding=padding)

    print(f"Input shape: {input_shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"Bias shape: {bias_shape}")
    print(f"Stride: {stride}")
    print(f"Padding: {padding}")
    print(f"Dilation: {dilation}")
    print(f"Groups: {groups}")
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

# Test the Conv3d operation
input_shape = (1, 1, 256, 256, 256)
weight_shape = (21, 1, 3, 3, 3)
bias_shape = (21,)
stride = 1
padding = 1
dilation = 1
groups = 1

test_conv3d(input_shape, weight_shape, bias_shape, stride, padding, dilation, groups)
