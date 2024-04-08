# %%
import torch
import torch.nn.functional as F
from aiimc.xbar import xLinear
from aiimc.xbar import xConv2d


# %%
# test the aiimc.xbar.xLinear


XbarSize = 4  # Crossbar size
in_features = 10
out_features = 2
batch_size = 2

model = xLinear(in_features, out_features, XbarSize=XbarSize)
x = torch.randn(batch_size, in_features)
y = model(x)
expected_result = F.linear(x, model.weight)


# Check if the expected and observed tensors are approximately equal
are_equal = torch.allclose(expected_result, y)

# Assert the result
assert are_equal, "Tensors are not equal"
print(f"The {type(model)} passted the Test!")

# %%
# test aiimc.xbar.xConv2d

# def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, XbarSize=256):
batch_size = 9
in_channels = 2
H = W = 10   
out_channels = 2
kernel_size = 3
XbarSize = 4  # Crossbar size
bias = False # test without bias. Bias is not implemented in xbar

model = xConv2d(in_channels, out_channels, kernel_size, bias=bias, XbarSize=256)
# modelT = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)

x = torch.rand(batch_size, in_channels, H, W)


expected_result = F.conv2d(x, model.weight.detach(), model.bias, model.stride,
                        model.padding, model.dilation, model.groups)

y = model(x)

# Check if the expected and observed tensors are approximately equal
are_equal = torch.allclose(expected_result, y, atol=1e-06) # absolute error tolerance is increased here!

# Assert the result
assert are_equal, "Tensors are not equal"
print(f"The {type(model)} passted the Test!")


