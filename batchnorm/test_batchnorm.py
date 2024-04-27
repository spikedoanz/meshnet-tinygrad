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

def test_batchnorm3d(input_shape, num_features):
    # Create numpy arrays for input
    input_np = np.random.randn(*input_shape).astype('float32')
    
    # Convert numpy array to PyTorch tensor
    input_tensor_torch = torch.from_numpy(input_np)
    bn_layer_torch = tnn.BatchNorm3d(num_features)
    bn_layer_torch.weight.data = torch.from_numpy(np.load("bn_weights.npy").astype("float32"))
    bn_layer_torch.bias.data = torch.from_numpy(np.load("bn_bias.npy").astype("float32"))
    output_torch = bn_layer_torch(input_tensor_torch)
    
    # tinygrad
    input_tensor_tg = Tensor(input_np)
    bn_layer_tg = BatchNorm3d(num_features)
    bn_layer_tg.weight = Tensor(np.load("bn_weights.npy"))
    bn_layer_tg.bias = Tensor(np.load("bn_bias.npy"))
    #bn_layer_tg.load_weights("bn_weights.npy", "bn_bias.npy")
    output_tg = bn_layer_tg(input_tensor_tg)

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

# Test the BatchNorm3d layer
input_shape = (1, 21, 256, 256, 256)
num_features = 21

test_batchnorm3d(input_shape, num_features)
