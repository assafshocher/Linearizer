import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from linearizer import LinearModule


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def batch_depthwise_conv2d(g_x, A, stride=1, padding='same', dilation=1):
    B, C, H, W = g_x.shape
    Kh, Kw = A.shape[-2:]

    x_ = g_x.reshape(1, B * C, H, W)  # (1, B*C, H, W)
    w_ = A.reshape(B * C, 1, Kh, Kw)  # (B*C, 1, Kh, Kw)

    y = F.conv2d(x_, w_, bias=None, stride=stride, padding=padding,
                 dilation=dilation, groups=B * C)  # (1, B*C, H, W) with 'same' padding
    y = y.reshape(B, C, y.shape[-2], y.shape[-1])  # (B, C, H, W)

    return y


class LinearKernel(LinearModule):
    def __init__(self, t_size=256, k=4):
        super().__init__()
        time_dim = t_size * k
        self.lin1 = nn.Sequential(
            SinusoidalPosEmb(t_size),
            nn.Linear(t_size, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, k * k * 3),
        )
        self.k = k

    def forward(self, x, **kwargs):
        A = self.get_matrix(kwargs['t'])
        x_pred = batch_depthwise_conv2d(x, A)
        return x_pred

    def get_matrix(self, t):
        A1 = self.lin1(t)
        A = A1.reshape(A1.shape[0], 3, self.k, self.k)
        return A
