from tinygrad import Tensor
from tinygrad import nn
from tinygrad import dtypes
import numpy as np

def create_random_kernel(in_channels, out_channels, k):
    return Tensor(np.random.randn(out_channels, in_channels, k, k, k).astype(np.float16))

channels = 5
kernel1 = create_random_kernel(1, channels, 3)
kernel2 = create_random_kernel(channels, channels, 3)
kernel3 = create_random_kernel(channels, channels, 3)
kernel4 = create_random_kernel(channels, channels, 3)
kernel5 = create_random_kernel(channels, channels, 3)
kernel6 = create_random_kernel(channels, channels, 3)
kernel7 = create_random_kernel(channels, channels, 3)
kernel8 = create_random_kernel(channels, channels, 3)
kernel9 = create_random_kernel(channels, channels, 3)
kernel10 = create_random_kernel(channels, 3, 1)

for _ in range(10):
    x0 = Tensor(np.random.randn(1, 1, 256, 256, 256).astype(np.float16))

    x1 = x0.conv2d(kernel1, stride=1, dilation=1, padding=1).relu()#.realize()

    # x2 = x1.conv2d(kernel2, stride=1, dilation=2, padding=2).relu()#.realize()
    #
    # x3 = x2.conv2d(kernel3, stride=1, dilation=4, padding=4).relu()#.realize()
    #
    # x4 = x3.conv2d(kernel4, stride=1, dilation=8, padding=8).relu()#.realize()
    #
    # x5 = x4.conv2d(kernel5, stride=1, dilation=16, padding=16).relu()#.realize()
    #
    # x6 = x5.conv2d(kernel6, stride=1, dilation=8, padding=8).relu()#.realize()
    #
    # x7 = x6.conv2d(kernel7, stride=1, dilation=4, padding=4).relu()#.realize()
    #
    # x8 = x7.conv2d(kernel8, stride=1, dilation=2, padding=2).relu()#.realize()
    #
    # x9 = x8.conv2d(kernel9, stride=1, dilation=1, padding=1).relu()#.realize()
    #
    # x10 = x9.conv2d(kernel10, stride=1, dilation=1, padding=0)#.realize()

x1.realize()
