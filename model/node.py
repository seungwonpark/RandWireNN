import torch
import torch.nn as nn
import torch.nn.functional as F

from .sep_conv import SeparableConv2d


class NodeOp(nn.Module):
    def __init__(self, in_degree, in_channel, out_channel, stride):
        super(NodeOp, self).__init__()
        self.agg_weight = nn.Parameter(torch.zeros(in_degree, requires_grad=True))
        self.conv = SeparableConv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, y):
        # y: [B, C, N, M, in_degree]
        y = torch.matmul(y, torch.sigmoid(self.agg_weight)) # [B, C, N, M]
        y = F.relu(y) # [B, C, N, M]
        y = self.conv(y) # [B, C_out, N, M]
        y = self.bn(y) # [B, C_out, N, M]
        return y


# if __name__ == '__main__':
#     x = torch.randn(7, 3, 224, 224, 5)
#     node = NodeOp(5, 3, 4)
#     y = node(x)
#     print(y.shape) # [7, 224, 224]
