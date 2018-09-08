import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):

    def __init__(self, in_channels, scale_init):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale_init = scale_init
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.in_channels))       
        self.init_weight()
 
    def init_weight(self):
        torch.nn.init.constant_(self.weight, self.scale_init)
    
    def forward(self, input):
        norm = input.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        norm_input = torch.div(input, norm)
        out = self.weight.view(1, -1, 1, 1).expand_as(input) * norm_input
        return out