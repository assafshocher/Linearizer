from typing import override

from linearizer.linearizer import Linearizer, G


class StyleLinearizer(Linearizer):
    def __init__(self, gx: G, linear_network, gy: G = None):
        super().__init__(gx=gx, linear_network=linear_network, gy=gy)

    @override
    def gx(self, x, **kwargs):
        return self.net_gx(x, mode='gx')

    @override
    def gy(self, y, **kwargs):
        return self.net_gy(y, mode='gy')

    @override
    def gx_inverse(self, g_x, **kwargs):
        return self.net_gx.inverse(g_x, mode='gx')

    @override
    def gy_inverse(self, g_y, **kwargs):
        return self.net_gy.inverse(g_y, mode='gy')
