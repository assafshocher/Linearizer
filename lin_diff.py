import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import InvTransformerNet
from torchvision.utils import make_grid
from utils import imwrite, find_latest_checkpoint
import wandb
import matplotlib.pyplot as plt


class LinearDiffusion(nn.Module):
    def __init__(self, conf, g=None):
        super().__init__()
        self.conf = conf
        # Ensure im_shape is a tuple.
        if not isinstance(self.conf.im_shape, tuple):
            self.conf.im_shape = tuple(self.conf.im_shape)
        im_sz = self.conf.im_shape[-3] * self.conf.im_shape[-2] * self.conf.im_shape[-1]
        self.rgb = (self.conf.im_shape[-3] == 3 )
    
        if g is None:
            self.g = InvTransformerNet(conf.n_heads, conf.n_layers, conf.p_sz, self.conf.im_shape[-1], self.rgb)
        else:
            self.g = g
        self.a = nn.Parameter(torch.randn(im_sz))

        if conf.flow_type is not None:
            self.calc_and_update_p_q_sigma(conf.T, conf.flow_type)

        # For model testing properties.
        self.gx = self.gy = self.g
        self.log_counter = 0


    def A(self):
        # Straight-through estimator for the binary mask A.
        a = self.a
        # A = (a > 0).float()
        # A = A.detach() + a - a.detach()
        A = (a.tanh() + 1)/2
        return A[None, :]
    
    def forward(self, x, t):
        # Surprisingly, this is almost never used, as training and sampling are done differently.
        # For the final main methods of the class see self.sample and self.train_model.
        gx = self.g(x)
        At_gx = self.A() * gx.view(gx.shape[0], -1) / self.sqrt_bar_alpha[t][:, None]
        return self.g.inverse(At_gx.view_as(gx))
    
    def get_losses(self, img=None, eps=None):
        if img is None and eps is None:
            raise ValueError("At least one of img or eps must be provided.")
        img = torch.zeros((0, *img.shape[1:]), device=img.device) if img is None else img
        eps = torch.zeros((0, *eps.shape[1:]), device=eps.device) if eps is None else eps
        
        # apply g to both img and eps
        g_out = self.g(torch.cat((img, eps), dim=0))
        g_img, g_eps = torch.split(g_out, (img.shape[0], eps.shape[0]))
        
        loss_img = ((1 - self.A()) * g_img.view(img.shape[0], -1)).pow(2).mean()
        loss_eps = (self.A() * g_eps.view(eps.shape[0], -1)).pow(2).mean()
        loss = self.conf.img_w * loss_img + self.conf.eps_w * loss_eps

        return loss, loss_img, loss_eps
    
    def sample(self, b_sz=1, nograd=True):
        if self.conf.sample_iterative:
            return self.sample_iterative(b_sz, nograd)
        
        with torch.no_grad() if nograd else torch.enable_grad():
            T = self.conf.T
            A = self.A()[0].bool()
            eps = torch.randn(b_sz*(T+1), *self.conf.im_shape, device=self.a.device)
            g_eps = self.g(eps).view(b_sz, -1, T+1)
            P, Q, Sigma = self.P, self.Q, self.Sigma
            g_x0 = ((P * Sigma).view(1, 1, -1) * g_eps).sum(-1)
            g_x0[:, A] += ((Q * Sigma).view(1, 1, -1) * g_eps[:, A, :]).sum(-1)
            g_x0 = g_x0.view(b_sz, *self.conf.im_shape)
            self.eval()
            x_0 = self.g.inverse(g_x0)
            self.train()
        return x_0
    

    def sample_iterative(self, b_sz=1, nograd=True):
        with torch.no_grad() if nograd else torch.enable_grad():
            T = self.conf.T
            A = self.A()[0] #.bool()
            x = torch.randn(b_sz, *self.conf.im_shape, device=self.a.device)
            a = self.a_forward
            b = self.b_forward
            sigma = self.Sigma
            print(f"Sampling iteratively with T={T}")

            self.eval()
            g_x = self.g(x)
            g_epses = torch.randn(T, b_sz, *self.conf.im_shape, device=self.a.device)
            gmin = []
            for t, g_eps in zip(range(T-2, 0, -1), g_epses):
                gmin.append(g_x.min().item())
                print("g_min: ", g_x.min().item(), "g_max: ", g_x.max().item(), "g_abs_min: ", g_x.abs().min().item())
                print("at: ", a[t].item(), "bt: ", b[t].item(), "sigma t: ", sigma[t].item())
                print("g_eps_min: ", g_eps.min().item(), "g_eps_max: ", g_eps.max().item(), "g_eps_abs_min: ", g_eps.abs().min())
                print("Unclear calculation: ", (a[t] + b[t] * A) )
                # tmp.append(self.g.inverse(g_x.view_as(x))[0, :].norm().item())
                # print("New calculation: ", self.g.inverse(g_x.view_as(x))[0, :].norm().item())
                # tmp2.append(g_x[0, :].norm().item())
                # import pdb; pdb.set_trace()
                g_x = (a[t] + b[t] * A) * g_x.view(g_x.shape[0], -1) + sigma[t] * self.g(g_eps).view(g_eps.shape[0], -1)
            x_0 = self.g.inverse(g_x.view_as(x))

        # if torch.isnan(x_0).any():
        #     import pdb; pdb.set_trace()
        self.train()
        return x_0

    def sample_g_xt(self, x_0, t):
        with torch.no_grad():
            return (self.sqrt_bar_alpha[t] * self.g(x_0).view(x_0.shape[0], -1) + 
                    torch.sqrt(1-self.bar_alpha[t]) * self.g(torch.randn_like(x_0)).view(x_0.shape[0], -1)).view_as(x_0)

                    
    
    def train_step(self, img, eps):
        loss, loss_img, loss_eps = self.get_losses(img, eps)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item(), loss_img.item(), loss_eps.item()
    
    def train_model(self, train_loader, n_epochs, val_loader=None):
        self.opt = optim.Adam(self.parameters(), lr=self.conf.lr)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, n_epochs)
        device = self.conf.device
        
        for epoch in range(n_epochs):
            running_loss = 0.0
            running_loss_img = 0.0
            running_loss_eps = 0.0
            num_batches = len(train_loader)
            for batch_idx, img in enumerate(train_loader):
                img = img[0].to(device)
                # Generate a batch of noise matching the shape of the images.
                eps = torch.randn_like(img)
                loss, loss_img, loss_eps = self.train_step(img, eps)
                running_loss += loss
                running_loss_img += loss_img
                running_loss_eps += loss_eps
                
                if (batch_idx + 1) % self.conf.log_freq == 0:
                    current_lr = self.opt.param_groups[0]['lr']
                    print(f"[Train] Epoch [{epoch+1}/{n_epochs}] Batch [{batch_idx+1}/{num_batches}] | "
                          f"LR: {current_lr:.6f} | Loss: {loss:.4f} (img: {loss_img:.4f}, eps: {loss_eps:.4f})")
                    if self.conf.wandb:
                        wandb.log({'epoch': epoch, 'batch': batch_idx, 'LR': current_lr,
                                   'loss': loss, 'loss_img': loss_img, 'loss_eps': loss_eps,
                                   'Arank': self.A().sum().item()}, step=self.log_counter)
                        self.log_counter += 1

            
            avg_loss = running_loss / num_batches
            avg_loss_img = running_loss_img / num_batches
            avg_loss_eps = running_loss_eps / num_batches
            print(f"[Train] Epoch {epoch+1} completed: Avg Loss: {avg_loss:.4f} "
                  f"(img: {avg_loss_img:.4f}, eps: {avg_loss_eps:.4f})")
            
            # Validation: generate samples and save grid.
            if (batch_idx + 1) % self.conf.val_freq == 0:
                self.valid(epoch+1, img)

            self.sched.step()

    def valid(self, epoch, img=None):
        """Validation: Generate samples using the sample method and save a grid of generated images."""
        from models import test_model_properties  # Call model property tests.
        self.eval()
        test_model_properties(self)
        # Determine the number of samples to generate (default to 16 if not specified in conf)
        sample_bs = getattr(self.conf, "val_sample_bs", 16)
        generated = self.sample(b_sz=sample_bs, nograd=True)
        grid = make_grid(generated, nrow=int(sample_bs ** 0.5))
        grid_save_path = os.path.join(self.conf.grid_dir, f"e{epoch}.png")
        imwrite(grid, grid_save_path)
        # we want to make a picture from self.a and save it as well
        a_img = self.A().view(*self.conf.im_shape).repeat(3,1,1)
        a_save_path = os.path.join(self.conf.grid_dir, f"a_e{epoch}.png")
        imwrite(a_img, a_save_path, bounds=(0,1))
        print(f"[Validation] Generated sample grid saved to {grid_save_path}")
        print(f"images range: {generated.min().item()} - {generated.max().item()}")
        print(f"a range: {a_img.min().item()} - {a_img.max().item()}")

        if img is not None:
            img = img[:sample_bs]
            t = self.conf.T // 3
            g_noisy = self.sample_g_xt(img, t)
            g_denoised = self.A() * g_noisy.view(img.shape[0], -1) / self.sqrt_bar_alpha[t]
            g_denoised = g_denoised.view_as(g_noisy)
            x_denoised = self.g.inverse(g_denoised)
            x_noisy = self.g.inverse(g_noisy)
            denoised_save_path = os.path.join(self.conf.grid_dir, f"denoised_e{epoch}.png")
            imwrite(make_grid(x_denoised, nrow=int(img.shape[0] ** 0.5)), denoised_save_path, bounds=(0,1))

            noisy_save_path = os.path.join(self.conf.grid_dir, f"noisy_e{epoch}.png")
            imwrite(make_grid(x_noisy, nrow=int(img.shape[0] ** 0.5)), noisy_save_path, bounds=(0,1))

            if self.conf.wandb:
                wandb.log({'denoising MSE': torch.sum((x_denoised - img)**2).item(),
                           'generated min': generated.min().item(),
                           'generated max': generated.max().item()}, step=self.log_counter)
        if self.conf.save_val_ckpt:
            self.save_checkpoint(f"e{epoch}.pth")
        self.train()


    def calc_and_update_p_q_sigma(self, T, flow_type):
        conf = self.conf
        device = self.a.device

        if conf.betas is None and conf.beta_min is not None and conf.beta_max is not None:
            conf.betas = torch.linspace(conf.beta_min, conf.beta_max, conf.T, device=device)
        elif conf.betas is not None and (conf.beta_min is None or conf.beta_max is None):
            conf.beta_min = min(conf.betas)
            conf.beta_max = max(conf.betas)
            if conf.T is None:
                conf.T = len(conf.betas)
            elif len(conf.betas) != conf.T:
                raise ValueError("Length of betas must match T.")
        else:
            raise ValueError("Either betas or beta_min and beta_max must be provided.")

        if flow_type == "DDPM":
            betas = conf.betas            # shape: (T,)
            alphas = 1 - betas            # shape: (T,)
            bar_alpha = torch.cumprod(alphas, dim=0)  # shape: (T,)
            sqrt_bar_alpha = torch.sqrt(bar_alpha)      # shape: (T,)
            
            # We want to compute coefficients for t = 1, ..., T-1 (0-indexed: index 0 corresponds to time step 1)
            # So we slice: bar_alpha[t-1] --> bar_alpha[:-1] and beta_t, alpha_t, bar_alpha[t] --> use indices [1:]
            a = sqrt_bar_alpha[:-1] * betas[1:] / (1 - bar_alpha[1:])
            # b = alphas[1:].sqrt() * (1 - bar_alpha[:-1]) / (1 - bar_alpha[1:])
            # b = b / sqrt_bar_alpha[1:]
            sigma = torch.sqrt(betas[1:] * (1 - bar_alpha[:-1]) / (1 - bar_alpha[1:]))
            # sigma = torch.cat([torch.ones(1, device=device), sigma], dim=0)
            # New try
            b = sqrt_bar_alpha[:-1] * (1 - alphas[1:]) / (1 - bar_alpha[1:])
            b = b / sqrt_bar_alpha[1:]
        elif flow_type == "DDIM":
            pass

        elif flow_type == "Rectified Flow Matching":
            raise NotImplementedError("Rectified Flow Matching is not implemented yet.")
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")
        
        # P = torch.empty(T+1, device=device)
        # Q = torch.empty(T+1, device=device)
        # P[0] = 1.0
        # Q[0] = 1.0
        # for t in range(1, T+1):
        #     P[t] = P[t-1] * a[t-1]
        #     Q[t] = Q[t-1] * (a[t-1] + b[t-1])
        # P[0] = P[T]
        # Q[0] = Q[T]

        # self.register_buffer("P", P)
        # self.register_buffer("Q", Q)
        self.register_buffer("Sigma", sigma)
        self.register_buffer("sqrt_bar_alpha", sqrt_bar_alpha)
        self.register_buffer("bar_alpha", bar_alpha)
        self.register_buffer("a_forward", a)
        self.register_buffer("b_forward", b)

    def save_checkpoint(self, filename):
        from datetime import datetime
        ckpt_path = os.path.join(self.conf.ckpt_dir, filename)
        torch.save(self.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
    
    def load_checkpoint(self, ckpt_file=None):
        if ckpt_file is None:
            ckpt_file = find_latest_checkpoint(self.conf.ckpt_dir, self.conf.exp_name)
            if ckpt_file is None:
                print("No checkpoint found.")
                return
        else:
            ckpt_file = os.path.join(self.conf.ckpt_dir, ckpt_file)
        ckpt = torch.load(ckpt_file, map_location=self.conf.device)
        model_has_module = any(k.startswith("module.") for k in self.state_dict().keys())
        ckpt_has_module = any(k.startswith("module.") for k in ckpt.keys())
        if model_has_module and not ckpt_has_module:
            ckpt = {f"module.{k}": v for k, v in ckpt.items()}
        if not model_has_module and ckpt_has_module:
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        self.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {ckpt_file}")
