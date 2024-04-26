import numpy as np

# Define the shape of the input and the conv_kwargs
input_shape = (1, 1, 256, 256, 256)
conv_kwargs = {
    "in_channels": 1,
    "out_channels": 21,
    "kernel_size": 3,
    "padding": 1,
    "stride": 1,
    "dilation": 1,
}

# Generate fake weights for the convolutional layer
conv_weights_shape = (
    conv_kwargs["out_channels"],
    conv_kwargs["in_channels"],
    conv_kwargs["kernel_size"],
    conv_kwargs["kernel_size"],
    conv_kwargs["kernel_size"]
)
conv_weights = np.random.randn(*conv_weights_shape).astype('float32')

# Generate fake weights and biases for the batch normalization layer
bn_weights_shape = (conv_kwargs["out_channels"],)
bn_weights = np.random.randn(*bn_weights_shape).astype('float32')
bn_bias = np.random.randn(*bn_weights_shape).astype('float32')

# Save the generated weights as .npy files
np.save("conv_weights.npy", conv_weights)
np.save("bn_weights.npy", bn_weights)
np.save("bn_bias.npy", bn_bias)

print("Fake weights generated and saved as .npy files.")
