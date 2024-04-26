import numpy as np
from tinygrad import nn
from tinygrad.tensor import Tensor

class ConvBNReLU:
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation, dropout_p=0):
        self.conv = Tensor.relu 
        self.bn = Tensor.relu
        self.relu = Tensor.relu
        self.dropout = Tensor.relu 

    def __call__(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))

class MeshNet:
    def __init__(self, n_channels, n_classes, large=True, dropout_p=0):
        if large:
            params = MeshNet_68_kwargs
        else:
            params = MeshNet_38_or_64_kwargs

        params[0]["in_channels"] = n_channels
        params[-1]["out_channels"] = n_classes

        self.layers = [ConvBNReLU(dropout_p=dropout_p, **block_kwargs) for block_kwargs in params[:-1]]
        self.layers.append(nn.Conv2d(**params[-1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

MeshNet_38_or_64_kwargs = [
    {"in_channels": -1, "kernel_size": 3, "out_channels": 21, "padding": 1, "stride": 1, "dilation": 1},
    {"in_channels": 21, "kernel_size": 3, "out_channels": 21, "padding": 1, "stride": 1, "dilation": 1},
    {"in_channels": 21, "kernel_size": 3, "out_channels": 21, "padding": 1, "stride": 1, "dilation": 1},
    {"in_channels": 21, "kernel_size": 3, "out_channels": 21, "padding": 2, "stride": 1, "dilation": 2},
    {"in_channels": 21, "kernel_size": 3, "out_channels": 21, "padding": 4, "stride": 1, "dilation": 4},
    {"in_channels": 21, "kernel_size": 3, "out_channels": 21, "padding": 8, "stride": 1, "dilation": 8},
    {"in_channels": 21, "kernel_size": 3, "out_channels": 21, "padding": 1, "stride": 1, "dilation": 1},
    {"in_channels": 21, "kernel_size": 1, "out_channels": -1, "padding": 0, "stride": 1, "dilation": 1},
]

MeshNet_68_kwargs = [
    {"in_channels": -1, "kernel_size": 3, "out_channels": 71, "padding": 1, "stride": 1, "dilation": 1},
    {"in_channels": 71, "kernel_size": 3, "out_channels": 71, "padding": 1, "stride": 1, "dilation": 1},
    {"in_channels": 71, "kernel_size": 3, "out_channels": 71, "padding": 2, "stride": 1, "dilation": 2},
    {"in_channels": 71, "kernel_size": 3, "out_channels": 71, "padding": 4, "stride": 1, "dilation": 4},
    {"in_channels": 71, "kernel_size": 3, "out_channels": 71, "padding": 8, "stride": 1, "dilation": 8},
    {"in_channels": 71, "kernel_size": 3, "out_channels": 71, "padding": 16, "stride": 1, "dilation": 16},
    {"in_channels": 71, "kernel_size": 3, "out_channels": 71, "padding": 1, "stride": 1, "dilation": 1},
    {"in_channels": 71, "kernel_size": 1, "out_channels": -1, "padding": 0, "stride": 1, "dilation": 1},
]

if __name__ == "__main__":

    # Create a random input tensor of shape (1, 1, 256, 256, 256)
    input_tensor = Tensor(np.random.rand(1, 1, 256, 256, 256).astype(np.float32))

    # Initialize the MeshNet model with random weights
    n_channels = 1
    n_classes = 3
    large = True
    dropout_p = 0.1

    model = MeshNet(n_channels, n_classes, large, dropout_p)

    # Perform a random forward pass
    output = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)
