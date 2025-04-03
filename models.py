import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention as MHA


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
    def __init__(self, num_heads, num_layers, patch_sz, im_sz, rgb=True):
        super().__init__()
        dim = 3*patch_sz**2 if rgb else patch_sz**2
        self.blocks = nn.ModuleList([InvTransformerBlock(dim//2, num_heads, patch_sz, im_sz) for _ in range(num_layers)])
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
        X = self.fold(X.permute(0,2,1))
        return X

    def inverse(self, Y):
        Y = self.unfold(Y).permute(0,2,1)
        for block, p in zip(reversed(self.blocks), reversed(self.p)):
            Y_1, Y_2 = Y.chunk(2, dim=-1)
            Y_1, Y_2 = block.inverse(Y_1, Y_2)
            Y = torch.cat([Y_1, Y_2], dim=-1)
            Y = p.inverse(Y)
        Y = self.fold(Y.permute(0,2,1))
        return Y


class InvTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_sz=16, im_sz=256):
        super().__init__()
        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads)
        self.G = MLPSubblock(dim=dim, patch_sz=patch_sz, im_sz=im_sz)
        self.s = 1. #2 ** (-0.5)

    def forward(self, X_1, X_2):
        Y_1 = (X_1 + self.F(X_2)) * self.s
        Y_2 = (X_2 + self.G(Y_1)) * self.s
        return Y_1, Y_2

    def inverse(self, Y_1, Y_2):
        X_2 = (Y_2 / self.s - self.G(Y_1)) 
        X_1 = (Y_1 / self.s - self.F(X_2)) 
        return X_1, X_2


class MLPSubblock(nn.Module):
    def __init__(self, dim, patch_sz, im_sz, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=True),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim, bias=True))

    def forward(self, x):
        return self.norm2(self.mlp(self.norm1(x)))


class AttentionSubBlock(nn.Module):
    def __init__(self, dim, num_heads, expand_ratio=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim * expand_ratio, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = MHA(dim * expand_ratio, num_heads, batch_first=True, bias=True)
        self.expand = nn.Linear(dim, dim * expand_ratio, bias=True)
        self.shrink = nn.Linear(dim * expand_ratio, dim, bias=True)
        self.v_start = dim * 2
        self.s = expand_ratio ** (-0.5)

    def forward(self, x):
        # we want biases for q, k but not for v
        with torch.no_grad():
            self.attn.in_proj_bias[self.v_start:].zero_()
            self.attn.out_proj.bias.zero_()
        x = self.expand(x)
        x = self.norm1(x)

        x, _ = self.attn(x, x, x)
        x = self.shrink(x)
        x = self.norm2(x)
        return x


def test_model_properties(model, bsz=64, test_y=False):
    device = next(model.parameters()).device
    im_sz_x = model.conf.im_shape[-1]
    n_inp_chans = 3 if model.rgb else 1
    x1, x2 = torch.randn(2*bsz, n_inp_chans, im_sz_x, im_sz_x, device=device).chunk(2, dim=0)
    a1, a2 = torch.randn(2*bsz, 1, 1, 1, device=device).chunk(2, dim=0)
    x = x1
    zx = torch.randn_like(x)
    if test_y:
        dim_y = model.dim_y
        y = torch.randn(bsz, 100, device=device)
        zy = torch.randn(bsz, dim_y, device=device)
    else:
        y = zy = None
    

    invertability_test(x, zx, y, zy, model, test_y)
    linearity_test(x1, x2, a1, a2, model)
    unitarity_test(x, y, model, test_y)



@torch.no_grad()
def invertability_test(x, zx, y, zy, model, test_y, thr=1e-2):
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

    if test_y:
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

    print(f"X->Z->X: {xzx_ok} ({xzx})\nZ->X->Z: {zxz_ok} ({zxz})")
    if test_y:
        print(f"Y->Z->Y: {yzy_ok} ({yzy})\nZ->Y->Z: {zyz_ok} ({zyz})")


@torch.no_grad()
def linearity_test(x1, x2, a1, a2, model, thr=1e-4):
    # f(a1x1+a2x2)
    zx1, zx2 = model.gx(x1), model.gx(x2)
    zx_superpos = a1*zx1 + a2*zx2
    x_superpos = model.gx.inverse(zx_superpos)
    t = torch.randint(0, model.conf.T, (x1.size(0),), device=x1.device)
    f_superpos_x = model(x_superpos, t)

    # a1f(x1)+a2f(x2)
    y1, y2 = model(x1, t), model(x2, t)
    zy1, zy2 = model.gy(y1), model.gy(y2)
    zy_superpos = a1*zy1 + a2*zy2
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
def unitarity_test(x, y, model, test_y, thr=10):
    zx_norm = model.gx(x).flatten(start_dim=1).pow(2).sum(dim=-1, keepdim=True)
    x_norm = x.flatten(start_dim=1).pow(2).sum(dim=-1, keepdim=True)
    ratio_x = (zx_norm / x_norm).mean()
    ratio_x_ok = ratio_x < thr

    if test_y:
        zy_norm = model.gy(y).flatten(start_dim=1).pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.flatten(start_dim=1).pow(2).sum(dim=-1, keepdim=True)
        ratio_y = (zy_norm / y_norm).mean()
        ratio_y_ok = ratio_y < thr

    print(f"Unitarity test: X:{ratio_x_ok} ({ratio_x.item()})")
    if test_y:
        print(f"Unitarity test: Y:{ratio_y_ok} ({ratio_y.item()})")


class TimeMLPBlock(nn.Module):
    def __init__(self, T, im_sz, rank, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T+1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, im_sz*(rank*2+1)))
        self.T = T
        self.rank = rank
        self.im_sz = im_sz
    def forward(self, t, at):
        # Create timestep embedding
        device = self.net[0].weight.device
        t = torch.tensor(t, device=device).view(-1, 1)
        t_one_hot = F.one_hot(t, self.T+1).float()
        t_embed = t_one_hot.flip(1).cumsum(dim=1).flip(1)
        params = self.net(t_embed).squeeze(1)
        # params = params * at.view(-1, 1)
        s = params[:, :self.im_sz]
        u, v = params[:, self.im_sz:].chunk(2, dim=-1)
        return (u.view(t.shape[0], self.im_sz, self.rank), 
                v.view(t.shape[0], self.rank, self.im_sz),
                s)


class FactorizedLinearLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, u, v, s):
        # Flatten spatial dimensions
        flat_x = x.reshape(x.shape[0], -1)
        
        # Apply factorized transformation: UVx
        vx = torch.einsum("b t d, b d -> b t", v, flat_x)
        uvx = torch.einsum("b d t, b t -> b d", u, vx)
        # Skip connection
        transformed = uvx + s * flat_x
        
        # Restore original shape
        return transformed.view_as(x)


class FactorizedLinearNet(nn.Module):
    def __init__(self, conf, rank=None, im_sz=None):
        super().__init__()
        self.conf = conf
        rank = rank or conf.A_rank
        im_sz = im_sz or conf.im_shape[-1] * conf.im_shape[-2]
        self.linear_layer = FactorizedLinearLayer()
        self.t_net = TimeMLPBlock(conf.T+1, 
                                  im_sz,
                                  rank,
                                  conf.mlp_hidden_dim)
        
    def forward(self, gx_t, t, at):
        u, v, s = self.t_net(t, at)
        g_hat_x0 = self.linear_layer(gx_t, u, v, s)
        return g_hat_x0


class LinearUnet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        chans = conf.im_shape[-3]
        ranks = [conf.A_rank * (2 ** i) for i in range(conf.n_levels)]
        im_szs = [int(conf.im_shape[-1]**2 * (4 ** (-i))) for i in range(conf.n_levels)]
        
        for rank, im_sz in zip(ranks, im_szs):
            print(rank, im_sz)
            self.down_blocks.append(
                nn.Conv2d(chans, chans, kernel_size=6, stride=2, padding=2, bias=False))
            self.linear_layers.append(
                FactorizedLinearNet(conf, rank, im_sz))
            self.up_blocks.append(
                nn.ConvTranspose2d(chans, chans, kernel_size=2, stride=2, output_padding=0, bias=False))
            
    def forward(self, x, t, at):
        block_results_down, block_results_up = [x], []
        for downblock in self.down_blocks:
            x = downblock(x)
            block_results_down.append(x)
        for block_result, linear_layer in zip(block_results_down, self.linear_layers):
            block_results_up.append(linear_layer(block_result, t, at))
        x = block_results_up[-1]
        for block_result, upblock in zip(reversed(block_results_up[:-1]), self.up_blocks):
            x = upblock(x) + block_result
        return x
            




class UnetBlock(nn.Module):
    def __init__(self, conf, chans_in, chans_out, down1up0=True, ksz=6, stride=2, n_layers=1, norm=nn.InstanceNorm2d, activation=nn.GELU()):
        super().__init__()
        if down1up0:
            self.main_conv = nn.Conv2d(chans_in, chans_out, kernel_size=ksz, padding=(ksz-stride)//2, bias=True, stride=stride)
        else:
            self.main_conv = nn.ConvTranspose2d(chans_in, chans_out, kernel_size=ksz, bias=True, output_padding=(ksz-stride)//2, stride=stride)
        
        self.extra_convs = nn.ModuleList([
                nn.Conv2d(chans_out, chans_out, kernel_size=3, padding=1, bias=False)
                for _ in range(n_layers-1)
            ])
        self.activation = activation
        self.norm = norm(chans_out) if norm else nn.Identity()
    
    def forward(self, x):
        x = self.main_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        for extra_conv in self.extra_convs:
            x = extra_conv(x)
            x = self.activation(x)
        return x



class Unet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        
        self.entry_conv = nn.Conv2d(conf.im_shape[0], conf.base_chans, kernel_size=3, padding=1, bias=True)
        self.exit_conv = nn.Conv2d(conf.base_chans, conf.im_shape[0], kernel_size=3, padding=1, bias=True)

        for ind in range(conf.n_levels):
            chans_in = conf.base_chans * (2 ** ind)
            chans_out = conf.base_chans * (2 ** (ind+1))
            self.down_blocks.append(
                UnetBlock(conf, down1up0=True, chans_in=chans_in, chans_out=chans_out, 
                          ksz=6, stride=2, n_layers=2))
            
            chans_in = conf.base_chans * (2 ** (conf.n_levels - ind))
            chans_out = conf.base_chans * (2 ** (conf.n_levels - ind - 1))
            self.up_blocks.append(
                UnetBlock(conf, down1up0=False, chans_in=chans_in, chans_out=chans_out, 
                          ksz=2, stride=2, n_layers=2))
            
    def forward(self, x, debug=False):
        x = self.entry_conv(x)
        block_results = [x]
        for downblock in self.down_blocks:
            if debug:
                print(x.shape)
            x = downblock(x)
            block_results.append(x)
        x = block_results[-1]
        for block_result, upblock in zip(reversed(block_results[:-1]), self.up_blocks):
            x = upblock(x) + block_result
            if debug:
                print(x.shape)
        x = self.exit_conv(x)
        return x
            