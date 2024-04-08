import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'xLinear',
    'xConv2d',
    ]


def _xbarmatmul(input, weight, XbarSize):
    """
    Implements the 2d matrix vector multiplication in a crossbar array.
    """
    if input.size(1) <= XbarSize:
            # If input size is smaller than or equal to N, use standard linear layer
            return F.linear(input, weight)
    else:
        # Split input and weight matrices into smaller chunks
        num_splits = input.size(1) // XbarSize
        split_inputs = torch.chunk(input, num_splits, dim=1)
        split_weights = torch.chunk(weight, num_splits, dim=1)

        # Compute intermediate results for each split
        intermediate_results = []
        for i in range(num_splits):
            intermediate_results.append(F.linear(split_inputs[i], split_weights[i]))

        # Sum the intermediate results
        return torch.stack(intermediate_results).sum(dim=0)

    

class xLinear(nn.Linear):
    """
    Implements a crossbar combatible version of the nn.Linear [Applies a linear transformation 
    to the incoming data: :math:`y = xA^T + b`. ] 
    """


    def __init__(self, in_features, out_features, bias=True, XbarSize=256):
        super().__init__(in_features, out_features, bias)
        
        self.XbarSize = XbarSize

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).

        TODO: current implementation is without bias. Need fixing!    
        """
        return _xbarmatmul(x, self.weight, self.XbarSize)
    
        


class xConv2d(nn.Conv2d):
    """
    This class can be used just like torch.nn.Conv2d, but keep in mind that this implementation 
    might not be as efficient as the original one, especially for large inputs or kernels. 
    This is because the unfold operation can be memory-intensive, and the linear operation 
    does not take advantage of specific optimizations for convolution operations. Also, this 
    implementation does not support grouped convolutions. If you need these features, you should 
    use torch.nn.Conv2d or another specialized function.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, XbarSize=256):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.XbarSize = XbarSize

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, in_height, in_width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, out_height, out_width).

        TODO: current implementation is without bias. Need fixing!    
        """    

        # Unfold the input tensor into a 2D matrix for linear operation
        x_unf = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        
        # Reshape weight for linear operation
        weight = self.weight.view(self.weight.size(0), -1)
        
        # Perform linear operation
        out_unf = _xbarmatmul(x_unf.transpose(1, 2), weight, self.XbarSize)

        # Fold the output tensor back into the input shape
        out = F.fold(out_unf.transpose(1, 2), ((x.shape[2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)// self.stride[0]+1, (x.shape[3]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1) // self.stride[1]+1), (1,1))
        
        return out
