import numpy as np
import torch
import torch.nn as tnn

from tinygrad import Tensor
from tinygrad import nn

class Conv3d_Torch(tnn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv3d_Torch, self).__init__()
        self.conv = tnn.Conv3d(*args, **kwargs)

    def load_weights(self, conv_weights_file):
        conv_weights = np.load(conv_weights_file)
        self.conv.weight.data = torch.from_numpy(conv_weights)

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv3d_TG:
    def __init__(self, *args, **kwargs):
        self.weight = None 
        self.bias = None
        self.stride = kwargs["stride"]
        self.dilation = kwargs["dilation"]
        self.padding = kwargs["padding"]

    def load_weights(self, conv_weights_file):
        self.weights = Tensor(np.load(conv_weights_file))

    def __call__(self, x):
        assert self.weights, "Weights not initialized"
        return x.conv2d(self.weights,
                        stride = self.stride,
                        dilation = self.dilation,
                        padding = self.padding)


def test_conv3d(input_shape, conv_kwargs):
    # Create numpy array for input
    input_np = np.random.randn(*input_shape).astype('float32')

    # Convert numpy array to PyTorch tensor
    input_tensor_torch = torch.from_numpy(input_np)
    conv_layer_torch = Conv3d_Torch(**conv_kwargs)
    conv_layer_torch.load_weights("conv_weights.npy")
    output_torch = conv_layer_torch(input_tensor_torch)

    # TinyGrad
    input_tensor_tg = Tensor(input_np)
    conv_layer_tg = Conv3d_TG(**conv_kwargs)
    conv_layer_tg.load_weights("conv_weights.npy")
    output_tg = conv_layer_tg(input_tensor_tg)

    print(f"Input shape: {input_shape}")
    print(f"Conv kwargs: {conv_kwargs}")
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
conv_kwargs = {
    "in_channels": 1,
    "out_channels": 21,
    "kernel_size": 3,
    "padding": 1,
    "stride": 1,
    "dilation": 2,
    "bias": False,
}

test_conv3d(input_shape, conv_kwargs)
