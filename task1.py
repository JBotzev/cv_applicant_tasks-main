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
class CustomGroupedConv2D(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)

class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(GroupedConv2d, self).__init__()

        # Calculate number of channels per group
        self.groups = groups
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups

        # Create stack of 2D convolution layers
        self.convs = nn.ModuleList()
        for i in range(groups):
            self.convs.append(nn.Conv2d(
                self.in_channels_per_group,
                self.out_channels_per_group,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=bias
            ))

    def forward(self, x):
        # Split input tensor along the channel dimension
        x_groups = torch.split(x, self.in_channels_per_group, dim=1)

        # Apply each 2D convolution layer to its corresponding group
        conv_results = []
        for i in range(self.groups):
            conv_results.append(self.convs[i](x_groups[i]))

        # Concatenate the results along the channel dimension and return
        return torch.cat(conv_results, dim=1)

    def _initialize_weights(self):
        # Initialize weights of each 2D convolution layer in the stack
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            if conv.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(conv.bias, -bound, bound)




        
