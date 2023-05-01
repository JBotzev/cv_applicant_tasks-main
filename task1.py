"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

# now write your custom layer
class CustomGroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=16, bias=True):
        super().__init__()

        # Create a stack of 2D convolutions with groups=1
        self.conv_stack = nn.ModuleList()
        for _ in range(groups):
            self.conv_stack.append(nn.Conv2d(in_channels//groups, out_channels//groups, kernel_size, stride, padding, groups=1, bias=bias))

    def forward(self, x):
        # Split the input tensor across the channel dimension
        x_split = torch.split(x, x.shape[1]//len(self.conv_stack), dim=1)
        out_list = []
        # Forward pass for each conv layer in the stack
        for i in range(len(self.conv_stack)):
            out_list.append(self.conv_stack[i](x_split[i]))
        # Concatenate the output tensors along the channel dimension
        out = torch.cat(out_list, dim=1)
        return out

# Create an instance of the custom layer
grouped_layer_custom = CustomGroupedConv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# Initialize the weights with Xavier uniform and biases with zeros
for conv_layer in grouped_layer_custom.conv_stack:
    nn.init.xavier_uniform_(conv_layer.weight)
    nn.init.zeros_(conv_layer.bias)

# Forward pass
y_custom = grouped_layer_custom(x)
print('original: ')
print(y)

print('custom: ')
print(y_custom)

# Check if the outputs are equal
print(torch.allclose(y, y_custom))
