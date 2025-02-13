import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention as MHA


class InvMLP(nn.Module):
    def __init__(self, dim, n_layers):
        super(InvMLP, self).__init__()
        self.t1 = nn.ModuleList([])
        self.t2 = nn.ModuleList([])
        self.p = nn.ModuleList([])
        for _ in range(n_layers):
            self.t1.append(nn.Sequential(nn.Linear(dim//2, dim//2, bias=False), nn.Tanh()))
            self.t2.append(nn.Sequential(nn.Linear(dim//2, dim//2, bias=False), nn.Tanh()))
            self.p.append(InvertiblePermutation(dim))
        self.n_layers = n_layers
        self.s = 2 ** (-0.5)
        
    def forward(self, x):
        if self.n_layers == 0:
            return x
        for t1, t2, p,in zip(self.t1, self.t2, self.p):
            x = p(x)
            x1, x2 = x.split(x.shape[-1]//2, dim=-1)
            x2 = (x2 + t2(x1)) * self.s
            x1 = (x1 + t1(x2)) * self.s
            x = torch.cat([x1, x2], dim=-1)
        return x
    
    def inverse(self, y):
        if self.n_layers == 0:
            return y
        for t1, t2, p in reversed(list(zip(self.t1, self.t2, self.p))):
            y1, y2 = y.split(y.size(1)//2, dim=-1)
            y1 = y1 / self.s - t1(y2)
            y2 = y2 / self.s - t2(y1)
            y = torch.cat([y1, y2], dim=-1)
            y = p.inverse(y)
        return y


class InvertiblePermutation(nn.Module):
    def __init__(self, n, axis=1):
        super(InvertiblePermutation, self).__init__()
        perm = torch.randperm(n)
        self.register_buffer('perm', perm)
        inv_perm = torch.argsort(perm)
        self.register_buffer('inv_perm', inv_perm)
        self.axis = axis

    def forward(self, x):
        return torch.index_select(x, self.axis, self.perm)

    def inverse(self, y):
        return torch.index_select(y, self.axis, self.inv_perm)

        


class InvTransformerNet(nn.Module):
    def __init__(self, num_heads, num_layers, patch_sz, im_sz):
        super().__init__()
        dim = 3*patch_sz**2
        self.blocks = nn.ModuleList([InvTransformerBlock(dim//2, num_heads) for _ in range(num_layers)])
        self.p = nn.ModuleList([InvertiblePermutation(dim, axis=2) for _ in range(num_layers)])
        self.unfold = nn.Unfold(kernel_size=patch_sz, stride=patch_sz)
        self.fold = nn.Fold(output_size=(im_sz, im_sz), kernel_size=patch_sz, stride=patch_sz)

    def forward(self, X):
        X = self.unfold(X).permute(0,2,1)
        for block, p in zip(self.blocks, self.p):
            X = p(X)
            X_1, X_2 = X.chunk(2, -1)
            X_1, X_2 = block(X_1, X_2)
            X = torch.cat([X_1, X_2], dim=-1)
        return X

    def inverse(self, Y):
        for block, p in zip(reversed(self.blocks), reversed(self.p)):
            Y_1, Y_2 = Y.chunk(2, dim=-1)
            Y_1, Y_2 = block.inverse(Y_1, Y_2)
            Y = torch.cat([Y_1, Y_2], dim=-1)
            Y = p.inverse(Y)
        Y = self.fold(Y.permute(0,2,1))
        return Y


class InvTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads)
        self.G = MLPSubblock(dim=dim)
        self.s = 2 ** (-0.5)

    def forward(self, X_1, X_2):
        Y_1 = (X_1 + self.F(X_2)) * self.s
        Y_2 = (X_2 + self.G(Y_1)) * self.s
        return Y_1, Y_2

    def inverse(self, Y_1, Y_2):
        X_2 = (Y_2 / self.s - self.G(Y_1)) 
        X_1 = (Y_1 / self.s - self.F(X_2)) 
        return X_1, X_2


class MLPSubblock(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim, bias=False))

    def forward(self, x):
        return self.norm2(self.mlp(self.norm1(x))) * 0.1
        # return self.mlp(self.norm1(x))



class AttentionSubBlock(nn.Module):
    def __init__(self, dim, num_heads, expand_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim * expand_ratio, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = MHA(dim * expand_ratio, num_heads, batch_first=True, bias=True)
        # self.expand = nn.Linear(dim, dim * expand_ratio, bias=False)
        # self.shrink = nn.Linear(dim * expand_ratio, dim, bias=False)
        self.v_start = dim * 2
        self.s = expand_ratio ** (-0.5)
        # Orthogonal initialization for in_proj_weight (Q, K, V)
        if hasattr(self.attn, 'in_proj_weight'):
            nn.init.orthogonal_(self.attn.in_proj_weight)
        
        # If bias exists, initialize it to zero
        if hasattr(self.attn, 'in_proj_bias') and self.attn.in_proj_bias is not None:
            nn.init.constant_(self.attn.in_proj_bias, 0.0)
        
        # Orthogonal initialization for out_proj.weight
        if hasattr(self.attn, 'out_proj') and hasattr(self.attn.out_proj, 'weight'):
            nn.init.orthogonal_(self.attn.out_proj.weight)
        
        # If out_proj has bias, initialize it to zero
        if hasattr(self.attn.out_proj, 'bias') and self.attn.out_proj.bias is not None:
            nn.init.constant_(self.attn.out_proj.bias, 0.0)

    def forward(self, x):
        # we want biases for q, k but not for v
        with torch.no_grad():
            self.attn.in_proj_bias[self.v_start:].zero_()
            self.attn.out_proj.bias.zero_()
        # x = self.expand(x)
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        # x = self.shrink(x)
        x = self.norm2(x)
        return x*0.1


def test_model_properties(model, bsz=64):
    device = next(model.parameters()).device
    im_sz_x, patch_sz_x, dim_y = model.sizes 
    dim_x = 3 * patch_sz_x ** 2
    n_tokens_x = (im_sz_x // patch_sz_x) ** 2
    x1, x2 = torch.randn(2*bsz, 3, im_sz_x, im_sz_x, device=device).chunk(2, dim=0)
    a1, a2 = torch.randn(2*bsz, 1, 1, device=device).chunk(2, dim=0)
    x = x1
    zx = torch.randn(bsz, n_tokens_x, dim_x, device=device)
    y = torch.randn(bsz, 100, device=device)
    zy = torch.randn(bsz, dim_y, device=device)

    invertability_test(x, zx, y, zy, model)
    linearity_test(x1, x2, a1, a2, model)
    unitarity_test(x, y, model)



@torch.no_grad()
def invertability_test(x, zx, y, zy, model, thr=1e-2):
    # X->Z->X
    zx_ = model.gx(x)
    x_ = model.gx.inverse(zx_)
    xzx = torch.norm(x - x_).item()
    xzx_ok = xzx < thr
    
    # Z->X->Z
    x_ = model.gx.inverse(zx)
    zx_ = model.gx(x_)
    zxz = torch.norm(zx - zx_).item()
    zxz_ok = zxz < thr

    # Y->Z->Y
    zy_ = model.gy(y)
    y_ = model.gy.inverse(zy_)
    yzy = torch.norm(y - y_).item()
    yzy_ok = yzy < 1e-2

    # Z->Y->Z
    y_ = model.gy.inverse(zy)
    zy_ = model.gy(y_)
    zyz = torch.norm(zy - zy_).item()
    zyz_ok = zyz < thr

    print(f"X->Z->X: {xzx_ok} ({xzx})\nZ->X->Z: {zxz_ok} ({zxz})\nY->Z->Y: {yzy_ok} ({yzy})\nZ->Y->Z: {zyz_ok} ({zyz})")


@torch.no_grad()
def linearity_test(x1, x2, a1, a2, model, thr=1e-4):
    # f(a1x1+a2x2)
    zx1, zx2 = model.gx(x1), model.gx(x2)
    zx_superpos = a1*zx1 + a2*zx2
    x_superpos = model.gx.inverse(zx_superpos)
    f_superpos_x = model(x_superpos, interp=False)

    # a1f(x1)+a2f(x2)
    y1, y2 = model(x1, interp=False), model(x2, interp=False)
    zy1, zy2 = model.gy(y1), model.gy(y2)
    zy_superpos = a1[:,:,0]*zy1 + a2[:,:,0]*zy2
    superpos_f_x = model.gy.inverse(zy_superpos)

    linearity_dist = (f_superpos_x - superpos_f_x).abs().mean()
    linearity_ok = linearity_dist < thr

    print(f"Linearity test: {linearity_ok} ({linearity_dist.item()})")
    
    zero_x = model.gx(x1*0).abs().mean()
    zero_x_ok = zero_x < thr
    zero_y = model.gy(y1*0).abs().mean()
    zero_y_ok = zero_y < thr

    print(f"Zero input test: X:{zero_x_ok} ({zero_x.item()}),   Y:{zero_y_ok} ({zero_y.item()})")


@torch.no_grad()
def unitarity_test(x, y, model, thr=10):
    zx = model.gx(x)
    ratio_x = (zx/x.view_as(zx)).abs().mean()
    ratio_x_ok = ratio_x < thr

    zy = model.gy(y)
    ratio_y = (zy/y.view_as(zy)).abs().mean()
    ratio_y_ok = ratio_y < thr

    print(f"Unitarity test: X:{ratio_x_ok} ({ratio_x.item()}),   Y:{ratio_y_ok} ({ratio_y.item()})")

