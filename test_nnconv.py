import numpy as np
import torch
import torch.nn as tnn

from tinygrad import Tensor
from tinygrad import nn

class Conv3d_Torch(tnn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv3d_Torch, self).__init__()
        self.conv = tnn.Conv3d(*args, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv3d_TG:
    def __init__(self, *args, **kwargs):
        c0 = kwargs["in_channels"]
        c1 = kwargs["out_channels"]
        stride = kwargs.get("stride", 1)
        kernel_size = (kwargs["kernel_size"],) * 3
        dilation = (kwargs.get("dilation", 1),) * 3
        padding = (kwargs["padding"],) * 6

        self.conv = nn.Conv2d(c0, c1,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation,
                              bias=kwargs.get("bias", True))

    def load_weights(self, conv_weights_file):
        # Load weights from .npy file
        conv_weights = np.load(conv_weights_file)

        # Cast the loaded weights to float32
        conv_weights = conv_weights.astype(np.float32)

        # Assign loaded weights to the layer
        self.conv.weight = Tensor(conv_weights)

    def __call__(self, x):
        x = self.conv(x)
        return x

def test_conv3d(input_shape, conv_kwargs):
    # Create numpy array for input
    input_np = np.random.randn(*input_shape).astype('float32')

    # Convert numpy array to PyTorch tensor
    input_tensor_torch = torch.from_numpy(input_np).float()

    # Create the Conv3d layer in PyTorch
    conv_layer_torch = Conv3d_Torch(**conv_kwargs)

    # Compute the output using PyTorch
    output_torch = conv_layer_torch(input_tensor_torch)

    # TinyGrad
    # Create an instance of Conv3d_TG
    conv_layer_tg = Conv3d_TG(**conv_kwargs)

    # Load the weights from .npy file
    conv_layer_tg.load_weights("conv_weights.npy")

    # Convert the input numpy array to a TinyGrad tensor
    input_tensor_tg = Tensor(input_np)

    # Compute the output using TinyGrad
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
    "dilation": 1,
    "bias": True,
}

test_conv3d(input_shape, conv_kwargs)
