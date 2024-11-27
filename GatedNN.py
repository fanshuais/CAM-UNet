# 1.门控卷积的模块
import torch
from torch import nn


class Gated_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, rate=1, activation=nn.ELU()):
        super(Gated_Conv, self).__init__()
        padding = int(rate * (ksize - 1) / 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=rate)
        self.activation = activation

    def forward(self, x1, x2):
        x = x1+x2
        raw = self.conv(x)  # 接受所有的输入生成权重更加合理
        gate = torch.sigmoid(raw)     # 第一部分的权重值
        gate_1 = 1-gate               # 第二部分的权重值
        out1 = self.activation(x1)
        out2 = self.activation(x2)

        return torch.cat([out1 * gate, out2 * gate_1], dim=1)

