import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import InvTransformerNet, test_model_properties, FactorizedLinearNet, LinearUnet
from torchvision.utils import make_grid
from utils import imwrite, find_latest_checkpoint, add_singletons_as, save_visuals, sample_timesteps
from data import mnist_denormalize as denorm
# import wandb


class LinearDiffusion(nn.Module):
    def __init__(self, conf, g=None):
        super().__init__()
        self.conf = conf
        # Ensure im_shape is a tuple.
        if not isinstance(self.conf.im_shape, tuple):
            self.conf.im_shape = tuple(self.conf.im_shape)
        self.rgb = (self.conf.im_shape[-3] == 3 )
    
        if g is None:
            g = InvTransformerNet(conf.n_heads, conf.n_layers, conf.p_sz, self.conf.im_shape[-1], self.rgb)
        self.g = g
            
        # Initialize the linear UNet model directly as self.A
        self.A = FactorizedLinearNet(conf)

        if conf.flow_type is not None:
            self.calc_and_update_p_q_sigma(conf.T, conf.flow_type)

        # For model testing properties.
        self.gx = self.gy = self.g
        self.log_counter = 0

    def device(self):
        return next(self.parameters()).device
    
    def forward(self, xt, t):
        gx = self.g(xt)
        g_hat_x0 = self.A(gx, t, self.a[t])
        return self.g.inverse(g_hat_x0)
    
    def sample(self, b_sz=1, nograd=True, xT=None, capture_frames=False):
        if self.conf.sample_iterative:
            return self.sample_x0_iterative(b_sz, nograd, xT, capture_frames)
        with torch.no_grad() if nograd else torch.enable_grad():
            pass

    def sample_x0_iterative(self, b_sz=1, nograd=True, xT=None, capture_frames=False):
        T = self.conf.T
        a, b = self.a, self.b
        print(f"Sampling iteratively with T={T}")
        with torch.no_grad() if nograd else torch.enable_grad():
            self.eval()
            xT = torch.randn(b_sz, *self.conf.im_shape, device=self.device()) if xT is None else xT
            g_xT = self.g(xT)
            g_xt = g_xT

            if capture_frames:
                frames = []
                frame_interval = T // self.conf.num_frames
            
            for t in range(self.conf.T, -1, -1):
                g_hat_x0 = self.A(g_xt, t, self.a[t]).view_as(g_xt)
                g_hat_xT = (g_xt - self.a[t] * g_hat_x0) / self.b[t]
                g_hat_xtminus1 = self.sample_g_xt(g_hat_x0, t-1, g_hat_xT)
                
                
                
                # g_hat_xt = self.sample_g_xt(g_hat_x0, t, g_xT)
                # g_hat_xtminus1 = self.sample_g_xt(g_hat_x0, t-1, g_xT)
                
                # a = self.a[t-1] / self.a[t] if t < 0.95*T else 1.

                # g_xt_minus1 = g_hat_xtminus1 + a * (g_xt - g_hat_xt)
                g_xt = g_hat_xtminus1
                
                if capture_frames and (t % frame_interval == 0 or t < 20):
                    frames.append((g_xt, g_hat_x0, t))

            hat_x_0 = self.g.inverse(g_hat_x0)
        self.train()
        
        if capture_frames:
            return hat_x_0, frames
        else:
            return hat_x_0

    def sample_g_xt(self, g_x0, t, g_xT=None):
        with torch.no_grad():
            g_xT = self.g(torch.randn_like(g_x0)) if g_xT is None else g_xT
            a = add_singletons_as(self.a[t], g_x0, start_dim=1)
            b = add_singletons_as(self.b[t], g_xT, start_dim=1)
            return a * g_x0 + b * g_xT
    
    def train_step(self, x0, eps, t):
        if self.conf.detach_noising:
            with torch.no_grad():
                g_xt = self.sample_g_xt(self.g(x0), t)
                xt = self.g.inverse(g_xt)
            hat_x0 = self.forward(xt.detach(), t)
        else:
            g_xt = self.sample_g_xt(self.g(x0), t)
            g_hat_x0 = self.A(g_xt, t, self.a[t])
            hat_x0 = self.g.inverse(g_hat_x0)


        

        loss = (hat_x0 - x0).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def train_model(self, train_loader, n_epochs):
        self.opt = optim.Adam(self.parameters(), lr=self.conf.lr)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, n_epochs)
        device = self.conf.device

        for epoch in range(n_epochs):
            running_loss = 0.0
            num_batches = len(train_loader)
            for batch_idx, img in enumerate(train_loader):
                img = img[0].to(device)
                # Generate a batch of noise matching the shape of the images.
                eps = torch.randn_like(img)
                t = torch.randint(0, self.conf.T+1, (img.shape[0],), device=img.device)
                # t = t * 0 #(torch.rand_like(t.float()) > 0.1).int  ()
                # t = sample_timesteps(self.a, img.shape[0])
                loss = self.train_step(img, eps, t)
                running_loss += loss
                if (batch_idx + 1) % self.conf.log_freq == 0:
                    current_lr = self.opt.param_groups[0]['lr']
                    print(f"[Train] Epoch [{epoch+1}/{n_epochs}] Batch [{batch_idx+1}/{num_batches}] | "
                          f"LR: {current_lr:.6f} | Loss: {loss:.8f}")
                    if self.conf.wandb:
                        wandb.log({'epoch': epoch, 'batch': batch_idx, 'LR': current_lr,
                                   'loss': loss}, step=self.log_counter)
                        self.log_counter += 1

            
            avg_loss = running_loss / num_batches
            print(f"[Train] Epoch {epoch+1} completed: Avg Loss: {avg_loss:.4f}")
            
            # Validation: generate samples and save grid.
            if (batch_idx + 1) % self.conf.val_freq == 0:
                self.valid(epoch+1, img)

            self.sched.step()

    
    def valid(self, epoch, img=None):
        self.eval()
        T = self.conf.T
        test_model_properties(self)
        sample_bs = getattr(self.conf, "val_sample_bs", 16)
        t = T // 2
        x0 = img[:sample_bs]

        
        results = {"x0": img[:sample_bs],
                   "xT": torch.randn_like(x0)}
        generated, frames = self.sample(b_sz=sample_bs, xT=results["xT"], 
                                        capture_frames=True)
        results = results | {
            "g_x0": self.g(x0),
            "g_xT": self.g(results["xT"]),
            "g_xt": self.sample_g_xt(self.g(x0), t, self.g(results["xT"]))}
        results = results | {
            "A_g_x0": self.A(results["g_x0"], t, self.a[t]),
            "A_g_xt": self.A(results["g_xt"], t, self.a[t]),
            "A_g_xT": self.A(results["g_xT"], t, self.a[t])}
        results = results | {
            "xt": self.g.inverse(results["g_xt"]),
            "denoised_xt": self.g.inverse(results["A_g_xt"]),
            "denoised_x0": self.g.inverse(results["A_g_x0"]),
            "denoised_xT": self.g.inverse(results["A_g_xT"]),
            "generated": generated}

        save_visuals(results, frames, self.g.inverse, 
                     self.conf.grid_dir, epoch=epoch)

        if self.conf.save_val_ckpt:
            self.save_checkpoint(f"e{epoch}.pth")
        self.train()

    
    def calc_and_update_p_q_sigma(self, T, flow_type):
        if flow_type == "linear":
            b = torch.arange(T+1, device=self.device()) / T
            a = 1 - b
        elif flow_type == "sqrt":
            a_ = torch.arange(T+1, device=self.device()) / T
            a = (1 - a_).sqrt()
            b = a_.sqrt()
        elif flow_type == "sqr":
            a_ = torch.arange(T+1, device=self.device()) / T
            a = (1 - a_) ** 2
            b = a_ ** 2
        elif flow_type == "cosine":
            s = 0.008
            t = torch.arange(T+1, device=self.device())
            a = ((t/T + s) * torch.pi * 0.5 / (1+s)).cos()
            b = (1 - a**2).sqrt()
        else:
            raise ValueError(f"Invalid flow type: {flow_type}")
            
        self.register_buffer("a", a)
        self.register_buffer("b", b)

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

    