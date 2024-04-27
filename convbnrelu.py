import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F

from tinygrad import Tensor
from tinygrad import nn

# from https://github.com/tinygrad/tinygrad/issues/1318
class Conv3d_TG:
    def __init__(self, *args, **kwargs):
        self.weight = None 
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
                        padding = self.padding,)


# from https://github.com/neuroneural/brainchop/blob/master/py2tfjs/meshnet.py
""" Torch spec:
    def conv_w_bn_before_act(dropout_p=0, *args, **kwargs):
    "Configurable Conv block with Batchnorm and Dropout"
    return nn.Sequential(
        nn.Conv3d(*args, **kwargs),
        nn.BatchNorm3d(kwargs["out_channels"]),
        nn.ReLU(inplace=True),
        nn.Dropout3d(dropout_p),
    )
"""

class ConvBNReLU(tnn.Module):
    def __init__(self, dropout_p=0, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = tnn.Conv3d(*args, **kwargs, bias=False)
        self.bn = tnn.BatchNorm3d(kwargs["out_channels"])
        self.relu = tnn.ReLU(inplace=True)
        self.dropout = tnn.Dropout3d(dropout_p)

    def load_weights(self, conv_weights_file, bn_weights_file, bn_bias_file):
        # Conv 
        conv_weights = np.load(conv_weights_file)
        self.conv.weight.data = torch.from_numpy(conv_weights)


        # Batchnorm
        bn_weights = np.load(bn_weights_file)
        bn_bias = np.load(bn_bias_file)
        self.bn.weight.data = torch.from_numpy(bn_weights)
        self.bn.bias.data = torch.from_numpy(bn_bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        #x = self.dropout(x)
        return x


class ConvBNReLU_TG:
    def __init__(self, dropout_p=0, *args, **kwargs):
        self.conv = Conv3d_TG(**kwargs)
        self.batchnorm = nn.InstanceNorm(kwargs["out_channels"],affine=True)
        self.dropout_p = dropout_p

    def load_weights(self, conv_weights_file, bn_weights_file, bn_bias_file):
        # Conv 
        self.conv.load_weights(conv_weights_file)

        # Batchnorm
        bn_weights = np.load(bn_weights_file)
        bn_bias = np.load(bn_bias_file)

        self.batchnorm.weight = Tensor(bn_weights)
        self.batchnorm.bias = Tensor(bn_bias)

    def __call__(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = x.relu()
        #x = x.dropout(self.dropout_p)
        return x

def test_conv_w_bn_before_act(input_shape, conv_kwargs, dropout_p=0):
    # Create numpy arrays for input
    input_np = np.random.randn(*input_shape).astype('float32')
    
    # Convert numpy array to PyTorch tensor
    input_tensor_torch = torch.from_numpy(input_np)
    conv_layer_torch = ConvBNReLU(dropout_p, **conv_kwargs)
    conv_layer_torch.load_weights("conv_weights.npy", "bn_weights.npy", "bn_bias.npy")
    output_torch = conv_layer_torch(input_tensor_torch)
    
    # tinygrad
    input_tensor_tg = Tensor(input_np)
    conv_bn_relu_tg = ConvBNReLU_TG(dropout_p=dropout_p, **conv_kwargs)
    conv_bn_relu_tg.load_weights("conv_weights.npy", "bn_weights.npy", "bn_bias.npy")
    output_tg = conv_bn_relu_tg(input_tensor_tg)


    #print("PyTorch output:")
    #print(output_torch.detach().numpy())
    print(f"Output shape (PyTorch): {output_torch.shape}")
    print()
    
    print(f"Input shape: {input_shape}")
    print(f"Conv kwargs: {conv_kwargs}")
    print(f"Dropout probability: {dropout_p}")
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

# Test the conv_w_bn_before_act layer
input_shape = (1, 1, 256, 256, 256)
conv_kwargs = {
    "in_channels": 1,
    "out_channels": 21,
    "kernel_size": 3,
    "padding": 1,
    "stride": 1,
    "dilation": 1,
}
dropout_p = 0.1

test_conv_w_bn_before_act(input_shape, conv_kwargs, dropout_p)
