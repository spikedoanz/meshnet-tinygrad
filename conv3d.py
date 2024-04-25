import torch
import numpy as np
from tinygrad import Tensor
from tinygrad.dtype import DType
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict, DefaultDict, cast, get_args, Set
import torch.nn.functional as F

def test_conv3d(input_shape, weight_shape, stride, padding, dilation, groups):
    # Create numpy arrays for input, weight, and bias
    input_np = np.random.randn(*input_shape)
    weight_np = np.random.randn(*weight_shape)
    bias_np = np.random.randn(weight_shape[0])

    # Convert numpy arrays to PyTorch tensors
    input_tensor_torch = torch.from_numpy(input_np)
    weight_tensor_torch = torch.from_numpy(weight_np)
    bias_tensor_torch = torch.from_numpy(bias_np)

    # Convert numpy arrays to tinygrad tensors
    input_tensor_tinygrad = Tensor(input_np)
    weight_tensor_tinygrad = Tensor(weight_np)
    bias_tensor_tinygrad = Tensor(bias_np)

    # Compute conv3d using PyTorch
    output_torch = F.conv3d(input_tensor_torch, weight_tensor_torch, bias_tensor_torch, stride, padding, dilation, groups)

    # Compute conv3d using tinygrad
    output_tinygrad = conv3d(input_tensor_tinygrad, weight_tensor_tinygrad, bias_tensor_tinygrad, groups, stride, dilation, padding)

    print("PyTorch output:")
    print(output_torch.numpy())
    print(f"Output shape (PyTorch): {output_torch.shape}")
    print()

    print("tinygrad output:")
    print(output_tinygrad.numpy())
    print(f"Output shape (tinygrad): {output_tinygrad.shape}")
    print()

    print(f"Input shape: {input_shape}")
    print(f"Weight shape: {weight_shape}")
    print(f"Stride: {stride}")
    print(f"Padding: {padding}")
    print(f"Dilation: {dilation}")
    print(f"Groups: {groups}")
    print()

    # Compare the outputs of PyTorch and tinygrad
    if np.allclose(output_torch.numpy(), output_tinygrad.numpy()):
        print("PyTorch and tinygrad outputs are equal!")
    else:
        print("PyTorch and tinygrad outputs are NOT equal!")
    print()

def conv3d(x: Tensor, w: Tensor, b: Optional[Tensor] = None, groups: int = 1, stride: int = 1, dilation: int = 1, padding: int = 0) -> Tensor:
    assert groups == 1, "Grouped conv not supported"
    padding = [padding] * 6 if isinstance(padding, int) else padding
    x = pad3d(x, padding)
    
    bs, cin, _, _, _ = x.shape
    cout, _, kd, kh, kw = w.shape
    ys = tuple([(x.shape[i+2] - w.shape[i+2]) // stride + 1 for i in range(3)])
    
    y = Tensor.zeros((bs, cout, *ys))
    for i in range(ys[0]):
        for j in range(ys[1]):
            for k in range(ys[2]):
                x_window = x[:, :, i*stride:i*stride+kd, j*stride:j*stride+kh, k*stride:k*stride+kw]
                y[:, :, i, j, k] = (x_window.reshape(bs, cin, -1) * w.reshape(cout, cin, -1)).sum(-1)
    
    if b is not None:
        y += b.reshape(cout, 1, 1, 1)
    
    return y

def pad3d(x: Tensor, padding: Union[int, Tuple[int, int, int, int, int, int]]) -> Tensor:
    if isinstance(padding, int):
        padding = (padding,) * 6
    elif len(padding) == 3:
        padding = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    else:
        assert len(padding) == 6, "Invalid padding specification"

    px, py, pz = padding[:2], padding[2:4], padding[4:]
    shape = x.shape
    new_shape = (shape[0], shape[1],
                 shape[2] + px[0] + px[1],
                 shape[3] + py[0] + py[1],
                 shape[4] + pz[0] + pz[1])

    padded_tensor = Tensor.zeros(new_shape)
    padded_tensor[:, :,
                 px[0]:px[0]+shape[2],
                 py[0]:py[0]+shape[3],
                 pz[0]:pz[0]+shape[4]] = x

    return padded_tensor

testn = 1
# Test case 1: Basic convolution
if testn > 0:
    test_conv3d(
        input_shape=(1, 1, 4, 4, 4),
        weight_shape=(1, 1, 3, 3, 3),
        stride=1,
        padding=0,
        dilation=1,
        groups=1
    )

# Test case 2: Stride > 1
if testn > 1:
    test_conv3d(
        input_shape=(1, 1, 8, 8, 8),
        weight_shape=(1, 1, 3, 3, 3),
        stride=2,
        padding=0,
        dilation=1,
        groups=1
    )

# Test case 3: Padding > 0
if testn > 2:
    test_conv3d(
        input_shape=(1, 1, 5, 5, 5),
        weight_shape=(1, 1, 3, 3, 3),
        stride=1,
        padding=1,
        dilation=1,
        groups=1
    )

# Test case 4: Dilation > 1
if testn > 3:
    test_conv3d(
        input_shape=(1, 1, 7, 7, 7),
        weight_shape=(1, 1, 3, 3, 3),
        stride=1,
        padding=0,
        dilation=2,
        groups=1
    )

# Test case 5: Groups > 1
if testn > 4:
    test_conv3d(
        input_shape=(1, 2, 6, 6, 6),
        weight_shape=(4, 1, 3, 3, 3),
        stride=1,
        padding=0,
        dilation=1,
        groups=2
    )

# Test case 6: Batch size > 1
if testn > 5:
    test_conv3d(
        input_shape=(2, 1, 5, 5, 5),
        weight_shape=(1, 1, 3, 3, 3),
        stride=1,
        padding=0,
        dilation=1,
        groups=1
    )

# Test case 7: Multiple input and output channels
if testn > 6: 
    test_conv3d(
        input_shape=(1, 3, 6, 6, 6),
        weight_shape=(2, 3, 3, 3, 3),
        stride=1,
        padding=0,
        dilation=1,
        groups=1
    )


