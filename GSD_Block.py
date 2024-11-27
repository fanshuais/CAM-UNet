import torch
import torch.nn as nn
from .SE import SE_Block
from .GatedNN import Gated_Conv
from .SAM import SpatialAttention


class GCS_Block(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(GCS_Block, self).__init__()
        self.Conv2d = nn.Conv2d(in_ch*2, in_ch, kernel_size=3, stride=1, padding=1)
        self.SE_Block = SE_Block(in_ch)
        self.SAM_Block = SpatialAttention()
        self.C1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.DilatedCNN = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation,dilation=dilation)
        self.GNN = Gated_Conv(in_ch, out_ch)
        self.GNN1 = Gated_Conv(in_ch, out_ch)
        self.GNN2 = Gated_Conv(in_ch, out_ch)

    def forward(self, xj, xi):  # xi为合并块，xj为被合并块
        x = torch.cat([xj, xi], dim=1)  # 通道上合并，初始特征
        x = self.Conv2d(x)  # 3x3卷积
        x1 = self.SE_Block(x)   # 通道注意力
        x1 = self.C1(x1)    # 1x1 卷积
        x2 = self.SAM_Block(x)      # 空间注意力
        x2 = self.DilatedCNN(x2)    # 空洞卷积
        x_sc = self.GNN(x1, x2)        # 通道特征与空间特征的门控融合
        x_s = self.GNN1(x1, x)        # 通道特征与原始特征的融合
        x_c = self.GNN2(x2, x)         # 空间特征与原始特征的融合
        x = x_s + x_c + x_sc
        return x


