"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# write your code here ...
import onnxruntime
import torch.nn.functional as F
import onnx

# Set random seed
torch.manual_seed(8)
np.random.seed(8)

# Load the ONNX model
model = onnx.load("model/model.onnx")
# Print the onnx architecture
print(model.graph)
# Load the ONNX model using ONNX Runtime
session = onnxruntime.InferenceSession("model/model.onnx")

# Get the input and output names of the model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define the PyTorch model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.bn2 = nn.BatchNorm2d(64)

        self.linear1 = nn.Linear(64*8*8, 512)
        nn.init.normal_(self.linear1.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(512, 10)
        nn.init.normal_(self.linear2.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

# Create an instance of the PyTorch model
my_model = MyModel()
# Print the model architecture
print(my_model)
# Prepare the input data for the model
input_data = np.random.rand(1, 3, 160, 320).astype(np.float32)
# Run the model on the input data using ONNX Runtime
output = session.run([output_name], {input_name: input_data})[0]
# Print the output
print(output)
