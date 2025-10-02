from typing import override

import torch
import torch.nn as nn

from linearizer.linearizer import G


# ---------- Small helpers ----------

class ActNorm2d(nn.Module):
    """Per-channel affine normalization with data-dependent init."""

    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, C, 1, 1))
        self.initialized = False
        self.eps = eps

    @torch.no_grad()
    def _init(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True) + self.eps
        self.bias.data.copy_(-mean)
        self.log_scale.data.copy_(-torch.log(std))
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init(x)
        return (x + self.bias) * torch.exp(self.log_scale)

    def inverse(self, y):
        if not self.initialized:
            return y
        return y * torch.exp(-self.log_scale) - self.bias


class Inv1x1Conv(nn.Module):
    """Glow-style invertible 1x1 convolution (channel mixer)."""

    def __init__(self, C):
        super().__init__()
        W = torch.linalg.qr(torch.randn(C, C)).Q  # orthogonal init => invertible
        self.W = nn.Parameter(W)

    def forward(self, x):  # x: [B,C,H,W]
        return torch.einsum('oi,bihw->bohw', self.W, x)

    def inverse(self, y):
        W_inv = torch.linalg.inv(self.W)
        return torch.einsum('oi,bihw->bohw', W_inv, y)


class Squeeze2x2(nn.Module):
    """Space-to-depth (forward) and depth-to-space (inverse). Requires even H,W."""

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for Squeeze2x2."
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()  # [B,C,2,2,H/2,W/2]
        x = x.view(B, C * 4, H // 2, W // 2)
        return x

    def inverse(self, y):
        B, C, H, W = y.shape
        assert C % 4 == 0, "Channels must be divisible by 4 for inverse Squeeze2x2."
        y = y.view(B, C // 4, 2, 2, H, W)
        y = y.permute(0, 1, 4, 2, 5, 3).contiguous()  # [B,C/4,H,2,W,2]
        y = y.view(B, C // 4, H * 2, W * 2)
        return y


# Wrap your creat_song_unet into a simple module that returns 2*C_half channels
class SongUNetWrapper(nn.Module):
    def __init__(self, base_unet):
        super().__init__()
        self.net = base_unet

    def forward(self, x, t):
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)
        return self.net(x, t, None)


class AffineCoupling(nn.Module):
    """
    y1 = (x1 + shift(x2)) * exp(clamp(log_s(x2)))
    inverse: x1 = y1 * exp(-clamp(log_s(x2))) - shift(x2)
    """

    def __init__(self, C_half, conditioner: nn.Module, clamp: float = 5.0):
        super().__init__()
        self.cond = conditioner  # outputs 2*C_half channels
        self.C_half = C_half
        self.clamp = clamp

    def _shift_log_scale(self, x_cond, t):
        h = self.cond(x_cond, t)  # [B, 2*C_half, H, W]
        shift, log_s = torch.chunk(h, 2, dim=1)
        log_s = torch.clamp(log_s, -self.clamp, self.clamp)
        return shift, log_s

    def forward(self, x1, x2, t):
        shift, log_s = self._shift_log_scale(x2, t)
        return (x1 + shift) * torch.exp(log_s)

    def inverse(self, y1, x2, t):
        shift, log_s = self._shift_log_scale(x2, t)
        return y1 * torch.exp(-log_s) - shift


# ---------- The improved invertible blocks you asked for ----------

class InvUnetBlock(nn.Module):
    """
    One invertible block on tensors with even channel count C:
      ActNorm -> Affine coupling (x1|x2) -> Affine coupling (x2|y1) -> invertible 1x1 conv
    """

    def __init__(self, C: int, img_resolution: int,
                 creat_song_unet,  # factory: (model_channels, in_channels, out_channels, img_resolution)
                 model_channels: int = 64,
                 use_actnorm: bool = True,
                 clamp: float = 5.0):
        super().__init__()
        assert C % 2 == 0, "InvUnetBlock requires even channels."
        C_half = C // 2

        F_cond = creat_song_unet(model_channels=model_channels,
                                 in_channels=C_half,
                                 out_channels=2 * C_half,
                                 img_resolution=img_resolution,
                                 channel_mult=[1, 1],
                                 num_blocks=2
                                 )
        G_cond = creat_song_unet(model_channels=model_channels,
                                 in_channels=C_half,
                                 out_channels=2 * C_half,
                                 img_resolution=img_resolution,
                                 channel_mult=[1, 1],
                                 num_blocks=2)

        self.F = AffineCoupling(C_half, SongUNetWrapper(F_cond), clamp=clamp)
        self.G = AffineCoupling(C_half, SongUNetWrapper(G_cond), clamp=clamp)

        self.mix = Inv1x1Conv(C)
        self.actnorm = ActNorm2d(C) if use_actnorm else None

    def forward(self, X, t):  # [B,C,H,W]
        if self.actnorm is not None:
            X = self.actnorm(X)
        x1, x2 = X.chunk(2, dim=1)
        y1 = self.F(x1, x2, t)
        y2 = self.G(x2, y1, t)
        Y = torch.cat([y1, y2], dim=1)
        Y = self.mix(Y)
        return Y

    def inverse(self, Y, t):
        Y = self.mix.inverse(Y)
        y1, y2 = Y.chunk(2, dim=1)
        x2 = self.G.inverse(y2, y1, t)
        x1 = self.F.inverse(y1, x2, t)
        X = torch.cat([y1 := x1, y2 := x2], dim=1)  # just to keep names aligned
        if self.actnorm is not None:
            X = self.actnorm.inverse(X)
        return X


class InvUnet(nn.Module):
    """
    Works with 1-channel inputs by squeezing to 4 channels, processing with blocks (C_eff),
    then unsqueezing back to 1 channel. For C>1, squeeze is optional but often helpful.
    """

    def __init__(self, num_layers: int,
                 in_channels: int,
                 img_resolution: int,
                 creat_song_unet,
                 model_channels: int = 16,
                 use_actnorm: bool = True,
                 clamp: float = 5.0,
                 use_squeeze_when_1ch: bool = True):
        super().__init__()

        self.use_squeeze = (in_channels % 2 == 1) and use_squeeze_when_1ch
        self.squeeze = Squeeze2x2() if self.use_squeeze else None

        C_eff = in_channels * (4 if self.use_squeeze else 1)
        assert C_eff % 2 == 0, "Effective channels must be even (squeeze ensures this for 1ch)."

        self.blocks = nn.ModuleList([
            InvUnetBlock(C=C_eff,
                         img_resolution=(img_resolution // 2 if self.use_squeeze else img_resolution),
                         creat_song_unet=creat_song_unet,
                         model_channels=model_channels,
                         use_actnorm=use_actnorm,
                         clamp=clamp)
            for _ in range(num_layers)
        ])

    def forward(self, X, t):  # X: [B, C_in, H, W]
        if self.use_squeeze:
            X = self.squeeze(X)  # [B, 4*C_in, H/2, W/2]
        for blk in self.blocks:
            X = blk(X, t)
        if self.use_squeeze:
            X = self.squeeze.inverse(X)
        return X

    def inverse(self, Y, t):  # Y: [B, C_in, H, W]
        if self.use_squeeze:
            Y = self.squeeze(Y)
        for blk in reversed(self.blocks):
            Y = blk.inverse(Y, t)
        if self.use_squeeze:
            Y = self.squeeze.inverse(Y)
        return Y


class InverseUnet(G):
    def __init__(self, num_of_layers, in_ch, img_resolution, creat_song_unet, model_channels=16):
        super().__init__(in_ch=in_ch, image_resolution=img_resolution)
        self.g = InvUnet(num_of_layers * 2, in_ch, img_resolution, creat_song_unet, model_channels=model_channels)

    def forward(self, x, **kwargs):
        if kwargs['mode'] == 'gy':
            return self.g(x, torch.zeros(x.shape[0], device=x.device))
        elif kwargs['mode'] == 'gx':
            return self.g(x, torch.ones(x.shape[0], device=x.device))
        else:
            raise NotImplementedError(f'No such mode exists: {kwargs['mode']}')

    def inverse(self, x, **kwargs):
        if kwargs['mode'] == 'gy':
            return self.g.inverse(x, torch.zeros(x.shape[0], device=x.device))
        elif kwargs['mode'] == 'gx':
            return self.g.inverse(x, torch.ones(x.shape[0], device=x.device))
        else:
            raise NotImplementedError(f'No such mode exists: {kwargs['mode']}')
