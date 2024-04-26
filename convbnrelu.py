import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F

from tinygrad import Tensor
from tinygrad import nn

# from https://github.com/tinygrad/tinygrad/issues/1318
class BatchNorm3d:
  def __init__(self, sz, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    if affine: self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
    else: self.weight, self.bias = None, None

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x:Tensor):
    # NOTE: this can be precomputed for static inference. we expand it here so it fuses
    batch_invstd = self.running_var.reshape(1, -1, 1, 1, 1).expand(x.shape).add(self.eps).rsqrt()
    bn_init = (x - self.running_mean.reshape(1, -1, 1, 1, 1).expand(x.shape)) * batch_invstd
    return self.weight.reshape(1, -1, 1, 1, 1).expand(x.shape) * bn_init + self.bias.reshape(1, -1, 1, 1, 1).expand(x.shape)

# from https://github.com/neuroneural/brainchop/blob/master/py2tfjs/meshnet.py
class ConvBNReLU(tnn.Module):
    def __init__(self, dropout_p=0, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = tnn.Conv3d(*args, **kwargs)
        self.bn = tnn.BatchNorm3d(kwargs["out_channels"])
        self.relu = tnn.ReLU(inplace=True)
        self.dropout = tnn.Dropout3d(dropout_p)

    def load_weights(self, conv_weights_file, bn_weights_file, bn_bias_file):
        # Load weights from .npy files
        conv_weights = np.load(conv_weights_file)
        bn_weights = np.load(bn_weights_file)
        bn_bias = np.load(bn_bias_file)

        # Cast the loaded weights to float32
        conv_weights = conv_weights.astype(np.float32)
        bn_weights = bn_weights.astype(np.float32)
        bn_bias = bn_bias.astype(np.float32)

        # Assign loaded weights to the layers
        self.conv.weight.data = torch.from_numpy(conv_weights)
        self.bn.weight.data = torch.from_numpy(bn_weights)
        self.bn.bias.data = torch.from_numpy(bn_bias)

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        return x

class ConvBNReLU_TG:
    def __init__(self, dropout_p=0, *args, **kwargs):
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
                              bias=False)
        self.batchnorm = BatchNorm3d(c1)
        self.relu = Tensor.relu
        self.dropout_p = dropout_p

    def load_weights(self, conv_weights_file, bn_weights_file, bn_bias_file):
        # Load weights from .npy files
        conv_weights = np.load(conv_weights_file)

        bn_weights = np.load(bn_weights_file)
        bn_bias = np.load(bn_bias_file)

        # Assign loaded weights to the layers
        self.conv.weight = Tensor(conv_weights)

        self.batchnorm.weight = Tensor(bn_weights)
        self.batchnorm.bias = Tensor(bn_bias)

    def __call__(self, x):
        x = self.conv(x)
        # x = self.batchnorm(x)
        # x = x.relu()
        # x = x.dropout(self.dropout_p)
        return x

def test_conv_w_bn_before_act(input_shape, conv_kwargs, dropout_p=0):
    # Create numpy arrays for input
    input_np = np.random.randn(*input_shape)
    
    # Convert numpy array to PyTorch tensor
    input_tensor_torch = torch.from_numpy(input_np).float()
    
    # Create the conv_w_bn_before_act layer in PyTorch
    conv_layer_torch = ConvBNReLU(dropout_p, **conv_kwargs)

    
    conv_layer_torch.load_weights("conv_weights.npy", "bn_weights.npy", "bn_bias.npy")
    # Compute the output using PyTorch
    output_torch = conv_layer_torch(input_tensor_torch)
    
    # tinygrad
    # Create an instance of ConvBNReLU_TG
    conv_bn_relu_tg = ConvBNReLU_TG(dropout_p=0.1, **conv_kwargs)

    # Load the weights from .npy files
    conv_bn_relu_tg.load_weights("conv_weights.npy", "bn_weights.npy", "bn_bias.npy")

    # Convert the input numpy array to a TinyGrad tensor
    input_tensor_tg = Tensor(input_np)

    # Use the ConvBNReLU_TG block in your model
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
