import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import InvTransformerNet
from torchvision.utils import make_grid
from utils import imsave, find_latest_checkpoint


class LinearDiffusion(nn.Module):
    def __init__(self, conf, g=None):
        super().__init__()
        self.conf = conf
        # Ensure im_shape is a tuple.
        if not isinstance(self.conf.im_shape, tuple):
            self.conf.im_shape = tuple(self.conf.im_shape)
        im_sz = self.conf.im_shape[-3] * self.conf.im_shape[-2] * self.conf.im_shape[-1]
        
        if g is None:
            self.g = InvTransformerNet(conf.n_heads, conf.n_layers, conf.p_sz, im_sz)
        else:
            self.g = g
        self.a = nn.Parameter(torch.randn(im_sz))

        if conf.flow_type is not None:
            self.calc_and_update_p_q_sigma(conf.T, conf.flow_type)

        # For model testing properties.
        self.gx = self.gy = self.g

    def A(self):
        a = self.a
        A = (a > 0).float()
        return A.detach() + a - a.detach()
    
    def forward(self, x, t):
        gx = self.g(x)
        At_gx = self.A * gx / self.conf.sqrt_bar_alpha[t]
        return self.g.inverse(At_gx)
    
    def get_losses(self, img=None, eps=None):
        if img is None and eps is None:
            raise ValueError("At least one of img or eps must be provided.")
        img = torch.zeros((0, *img.shape[1:]), device=img.device) if img is None else img
        eps = torch.zeros((0, *eps.shape[1:]), device=eps.device) if eps is None else eps
        
        # Flatten and concatenate.
        combined = torch.cat((img.view(-1), eps.view(-1)), dim=0)
        g_out = self.g(combined)
        # Split the output into two parts.
        g_img, g_eps = torch.split(g_out, img.numel())
        
        loss_img = ((1 - self.A) * g_img).mean()
        loss_eps = (self.A * g_eps).mean()
        loss = self.conf.img_w * loss_img + self.conf.eps_w * loss_eps

        return loss, loss_img, loss_eps
    
    def sample(self, b_sz=1, nograd=True):
        with torch.no_grad() if nograd else torch.enable_grad():
            T = self.conf.T
            A = self.A.bool()
            eps = torch.randn(b_sz, T+1, *self.conf.im_shape, device=self.a.device)
            g_eps = self.g(eps).view(b_sz, -1, 1)
            P, Q, Sigma = self.conf.P, self.conf.Q, self.conf.Sigma
            if not (P.shape[0] == Q.shape[0] == Sigma.shape[0]):
                raise ValueError("P, Q and Sigma must have the same length.")
            g_x0 = ((P * Sigma).view(1, 1, -1) * g_eps).sum(-1)
            g_x0[A] += ((Q[A] * Sigma[A]).view(1, 1, -1) * g_eps[A]).sum(-1)
            g_x0 = g_x0.view(b_sz, *self.conf.im_shape)
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
    
    def train_model(self, train_loader, n_epochs, val_loader=None):
        self.opt = optim.Adam(self.parameters(), lr=self.conf.lr)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, n_epochs)
        device = self.conf.device
        
        for epoch in range(n_epochs):
            running_loss = 0.0
            running_loss_img = 0.0
            running_loss_eps = 0.0
            num_batches = len(train_loader)
            for batch_idx, (img, eps) in enumerate(train_loader):
                img, eps = img.to(device), eps.to(device)
                loss, loss_img, loss_eps = self.train_step(img, eps)
                running_loss += loss
                running_loss_img += loss_img
                running_loss_eps += loss_eps
                
                if (batch_idx + 1) % self.conf.log_freq == 0:
                    current_lr = self.opt.param_groups[0]['lr']
                    print(f"[Train] Epoch [{epoch+1}/{n_epochs}] Batch [{batch_idx+1}/{num_batches}] | "
                          f"LR: {current_lr:.6f} | Loss: {loss:.4f} (img: {loss_img:.4f}, eps: {loss_eps:.4f})")
            
            avg_loss = running_loss / num_batches
            avg_loss_img = running_loss_img / num_batches
            avg_loss_eps = running_loss_eps / num_batches
            print(f"[Train] Epoch {epoch+1} completed: Avg Loss: {avg_loss:.4f} "
                  f"(img: {avg_loss_img:.4f}, eps: {avg_loss_eps:.4f})")
            
            if val_loader is not None:
                self.valid(val_loader, epoch+1)
    
    def valid(self, val_loader, epoch):
        """Validation on one batch; also calls test_model_properties."""
        from models import test_model_properties  # Call at every validation.
        self.eval()
        # Test model properties.
        test_model_properties(self)
        batch = next(iter(val_loader))
        img, eps = batch
        img, eps = img.to(self.conf.device), eps.to(self.conf.device)
        loss, loss_img, loss_eps = self.get_losses(img, eps)
        print(f"[Validation] Epoch {epoch} Loss: {loss.item():.4f} "
              f"(img: {loss_img.item():.4f}, eps: {loss_eps.item():.4f})")
        grid = make_grid(img, nrow=int(img.size(0) ** 0.5))
        grid_save_path = os.path.join(self.conf.grid_dir, f"grid_epoch_{epoch}_{datetime_now()}.png")
        imsave(grid, grid_save_path)
        if self.conf.save_val_ckpt:
            self.save_checkpoint(f"checkpoint_epoch_{epoch}_{datetime_now()}.pth")
        self.train()
    
    def calc_and_update_p_q_sigma(self, T, flow_type):
        conf = self.conf
        device = self.a.device
        one = torch.tensor(1.0, device=device)

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

        if flow_type in ["DDPM", "DDIM"]:
            alphas = 1 - conf.betas
            bar_alpha = torch.cumprod(alphas, dim=0)
            bar_alpha = torch.cat([one.unsqueeze(0), bar_alpha], dim=0)
            sqrt_bar_alpha = torch.sqrt(bar_alpha)
            a = torch.sqrt(bar_alpha[:-1] / bar_alpha[1:])
            b_old = (sqrt_bar_alpha[1:] - sqrt_bar_alpha[:-1]) / sqrt_bar_alpha[1:]
            b = b_old / sqrt_bar_alpha[1:]
            if flow_type == "DDPM":
                sigma = torch.sqrt(conf.betas)
                sigma = torch.cat([torch.ones(1, device=device), sigma], dim=0)
            else:
                sigma = torch.zeros(T+1, device=device)
                sigma[0] = 1.0
        elif flow_type == "Rectified Flow Matching":
            raise NotImplementedError("Rectified Flow Matching is not implemented yet.")
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")
        
        P = torch.empty(T+1, device=device)
        Q = torch.empty(T+1, device=device)
        P[0] = 1.0
        Q[0] = 1.0
        for t in range(1, T+1):
            P[t] = P[t-1] * a[t-1]
            Q[t] = Q[t-1] * (a[t-1] + b[t-1])
        P[0] = P[T]
        Q[0] = Q[T]

        conf.P = P
        conf.Q = Q
        conf.Sigma = sigma
        self.conf.sqrt_bar_alpha = sqrt_bar_alpha

    def save_checkpoint(self, filename=None):
        from datetime import datetime
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_{timestamp}.pth"
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
        # Adjust state dict keys for DDP if needed.
        model_has_module = any(k.startswith("module.") for k in self.state_dict().keys())
        ckpt_has_module = any(k.startswith("module.") for k in ckpt.keys())
        if model_has_module and not ckpt_has_module:
            ckpt = {f"module.{k}": v for k, v in ckpt.items()}
        if not model_has_module and ckpt_has_module:
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        self.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {ckpt_file}")

def datetime_now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")
