from abc import abstractmethod
from typing import final

import torch.nn


class G(torch.nn.Module):
    def __init__(self, in_ch, image_resolution):
        super().__init__()
        self.dim = in_ch * image_resolution ** 2

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def inverse(self, z, **kwargs):
        pass


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    # optional
    def inverse(self, z, **kwargs):
        pass


class Linearizer(torch.nn.Module):
    def __init__(self, gx: G, linear_network: LinearModule, gy: G = None):
        super().__init__()
        if gy is None:
            gy = gx
        self.net_gx = gx
        self.net_gy = gy
        self.linear_network = linear_network

    def gx(self, x, **kwargs):
        return self.net_gx(x, **kwargs)

    def gy(self, y, **kwargs):
        return self.net_gy(y, **kwargs)

    def gx_inverse(self, g_x, **kwargs):
        return self.net_gx.inverse(g_x, **kwargs)

    def gy_inverse(self, g_y, **kwargs):
        return self.net_gy.inverse(g_y, **kwargs)

    def A(self, g_x, **kwargs):
        return self.linear_network(g_x, **kwargs)

    # optional
    def A_inverse(self, g_y, **kwargs):
        return self.linear_network.inverse(g_y, **kwargs)

    @final
    def forward(self, x, **kwargs):
        g_x = self.gx(x, **kwargs)
        g_y = self.A(g_x, **kwargs)
        y_pred = self.gy_inverse(g_y, **kwargs)
        return y_pred

    @final
    def inverse(self, y, **kwargs):
        g_y = self.gy(y, **kwargs)
        g_x = self.A_inverse(g_y, **kwargs)
        x_pred = self.gx_inverse(g_x, **kwargs)
        return x_pred
