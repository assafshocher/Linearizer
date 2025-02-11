import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models_cifar import InvTransformerNet, test_model_properties
from data_cifar import get_data_loaders


class LinearDiffusion(nn.Module):
    def __init__(self, conf, g=None):
        super().__init__()
        self.conf = conf
        im_sz = conf.im_shape[-3] * conf.im_shape[-2] * conf.im_shape[-1]
        if g is None:
            self.g = InvTransformerNet(conf.n_heads, conf.n_layers, conf.p_sz, im_sz)
        self.a = nn.Parameter(torch.randn(im_sz))

        if conf.flow_type is not None:
            self.calc_and_update_p_q_sigma(conf.T, conf.flow_type)

    def A(self):
        a = self.a
        A = (a > 0).float()
        return A.detach() + a - a.detach()
    
    def forward(self, x, t):
        # This is applying f(x) denoising. We don't usually 
        # use it as trainnig and sampling are done differently.
        gx = self.g(x)
        At_gx = self.A * gx / self.conf.sqrt_bar_alpha[t]
        return self.g.inverse(At_gx)
    
    def get_losses(self, img=None, eps=None):
        if img is None and eps is None:
            raise ValueError("At least one of img or eps must be provided.")
        img = torch.zeros((0, *img.shape[1:]), device=img.device) if img is None else img
        eps = torch.zeros((0, *eps.shape[1:]), device=eps.device) if eps is None else eps
        
        g_img, g_eps = self.g(torch.cat((img.view(-1), eps.view(-1)), dim=0)).split(len(img))
        
        loss_img = ((1-self.A) * g_img).mean()
        loss_eps = (self.A * g_eps).mean()
        loss = self.conf.img_w*loss_img + self.conf.eps_w*loss_eps

        return loss, loss_img, loss_eps
    
    def sample(self, b_sz=1, nograd=True):
        with torch.no_grad() if nograd else torch.enable_grad():
            
            T = self.conf.T
            A = self.A.bool()

            # 1) Sample a batch of T+1, sized N IID random vectors from standard normal distribution.
            eps = torch.randn(b_sz, T+1, self.conf.im_shape[-3:], device=P.device)

            # 2) Apply g to the batch to get g(epsilon_t)_{0..T}.
            g_eps = self.g(eps).view(b_sz, -1, 1)

            # 3) Get P, Q, Sigma
            P, Q, Sigma = (self.conf.P,
                           self.conf.Q,
                           self.conf.Sigma)
            if not P.shape[0] == Q.shape[0] == Sigma.shape[0]:
                raise ValueError("P, Q and Sigma must have the same length.")

            # 4) Calculate g(x_0)
            g_x0 = ((P * Sigma).view(1, 1,-1) * g_eps).sum(-1)  # sum_1{[Nx1] @ [1xT]} -> [N]
            g_x0[A] += ((Q[A] * Sigma[A]).view(1, 1,-1) * g_eps[A]).sum(-1)
            g_x0 = g_x0.view(b_sz, *self.conf.im_shape[-3:])

            # 5) Apply g^-1 to calculate x_0
            self.eval()
            x_0 = self.g.inverse(g_x0)
            self.train()

        return x_0
    
    def train_step(self, img, eps):
        loss, loss_img, loss_eps = self.get_losses(img, eps)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.sched.step()
        return loss.item(), loss_img.item(), loss_eps.item()
    
    def train(self, data_loader, n_epochs, eps_bsz=None, valid=False):
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, n_epochs)

        device = self.a.device
        if eps_bsz is None:
            eps_bsz = data_loader.batch_size
        running_loss, running_loss_img, running_loss_eps = 0., 0., 0.

        for epoch in range(n_epochs):
            if epoch % self.conf.log_freq == 0:
                if valid:
                    self.valid()
                print(f"Epoch {epoch} | Loss: {running_loss:.4f} | Loss_img: {running_loss_img:.4f} | Loss_eps: {running_loss_eps:.4f}")

            running_loss, running_loss_img, running_loss_eps = 0., 0., 0.

            for img, eps in data_loader:
                img, eps = img.to(device), eps.to(device)
                loss, loss_img, loss_eps = self.train_step(img, eps)
                
                running_loss += loss
                running_loss_img += loss_img
                running_loss_eps += loss_eps

    def valid(self, b_sz):
        x0 = self.sample(b_sz)
        # TODO: save im, w&b, log, metrics, etconf.

    def calc_and_update_p_q_sigma(self, T, flow_type):
        conf = self.conf
        device = self.a.device  # ensure we work on the correct device
        one = torch.tensor(1.0, device=device)

        if conf.betas is None and conf.beta_min is not None and conf.beta_max is not None:
            # default schedule
            conf.betas = torch.linspace(conf.beta_min, 
                                        conf.beta_max,
                                        conf.T, 
                                        device=self.device)
        elif conf.betas is not None and (conf.beta_min is None or conf.beta_max is None):
            # custom schedule
            conf.beta_min = conf.betas.min().item()
            conf.beta_max = conf.betas.max().item()
            if conf.T is None:
                conf.T = len(conf.betas)
            elif len(conf.betas) != conf.T:
                raise ValueError("Length of betas must match T.")
        else:
            raise ValueError("Either betas or beta_min and beta_max must be provided.")

        if flow_type in ["DDPM", "DDIM"]:
            # x_t = sqrt(bar_alpha_t) * x0 + sqrt(1-bar_alpha_t) * eps,
            # where bar_alpha_t = prod_{s=1}^t (1 - beta_s).
            alphas = 1 - self.betas  # shape [T]
            bar_alpha = torch.cumprod(alphas, dim=0)  # shape [T]
            bar_alpha = torch.cat([one.unsqueeze(0), bar_alpha], dim=0)  # # Prepend bar_alpha_0 = 1.
            sqrt_bar_alpha = torch.sqrt(bar_alpha)  # This gives alpha_t for t = 0,...,T.
            # For t >= 1, define alpha_forward as:

            # Define the reverse process coefficients.
            # a_t = sqrt( bar_alpha[t-1] / bar_alpha[t] ) for t=1,...,T.
            a = torch.sqrt(bar_alpha[:-1] / bar_alpha[1:])  # length T
            # b_t^old = (sqrt_bar_alpha[t] - sqrt_bar_alpha[t-1]) / sqrt_bar_alpha[t] for t=1,...,T.
            b_old = (sqrt_bar_alpha[1:] - sqrt_bar_alpha[:-1]) / sqrt_bar_alpha[1:]
            # Now re-define b_t = b_old / alpha_forward.
            b = b_old / sqrt_bar_alpha[1:]  # length T

            if flow_type == "DDPM":
                # For DDPM, noise variance sigma_t = sqrt(beta[t]) for t>=1, and sigma_0 = 1.
                sigma = torch.sqrt(self.conf.betas)  # length T for t>=1.
                sigma = torch.cat([torch.ones(1, device=device), sigma], dim=0)  # now length T+1.
            else:  # DDIM
                # For DDIM, often sigma_t = 0 for t>=1, sigma_0 = 1.
                sigma = torch.zeros(T+1, device=device)
                sigma[0] = 1.0

        elif flow_type == "Rectified Flow Matching":
            pass  # TODO

        else:
            raise ValueError(f"Unknown flow type: {flow_type}")
        
        # Compute cumulative products P and Q for t=0,...,T.
        # Standardly, define for t>=1: P[t] = prod_{s=1}^{t} a_s and Q[t] = prod_{s=1}^{t} (a_s + b_s),
        # with the empty product defined as 1.
        P = torch.empty(T+1, device=device)
        Q = torch.empty(T+1, device=device)
        P[0] = 1.0
        Q[0] = 1.0
        for t in range(1, T+1):
            P[t] = P[t-1] * a[t-1]
            Q[t] = Q[t-1] * (a[t-1] + b[t-1])

        # Now apply the special convention from your derivation:
        # instead of P[0] = 1, we set P[0] = P[T], and similarly for Q.
        P[0] = P[T]
        Q[0] = Q[T]

        conf.P = P
        conf.Q = Q
        conf.Sigma = sigma
        self.sqrt_bar_alpha = sqrt_bar_alpha




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g. '0,1'). Use '' for CPU only.")
    parser.add_argument("--n_epochs", type=int, default=15000, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--log_freq", type=int, default=11, help="Log training loss every N batches.")
    parser.add_argument("--valid", action="store_true", default=True, help="whether to validate during training.")
    parser.add_argument("--save_freq", type=int, default=100, help="Save model checkpoint every N epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (for SGD).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 penalty).")
    parser.add_argument("--use_ddp", action="store_true", default=True,
                        help="Use DistributedDataParallel (DDP) instead of DataParallel or single-GPU.")

    # Model-specific arguments
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in g.")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of heads (for multi-head) in g.")
    parser.add_argument("--patch_sz", type=int, default=8, help="Patch size for g.")
    parser.add_argument("--im_shape", type=int, default=32, help="Image size for g.")


    

    conf = parser.parse_args()
    print(conf)