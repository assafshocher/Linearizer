import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import InvTransformerNet
from torchvision.utils import make_grid
from utils import imwrite, find_latest_checkpoint
from data import mnist_denormalize as denorm
# import wandb


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
        self.a_net = nn.Sequential(nn.Linear(conf.T+1, im_sz//4), nn.ReLU(), 
                                   nn.Linear(im_sz//4, im_sz//4), nn.ReLU(),
                                   nn.Linear(im_sz//4, im_sz))

        if conf.flow_type is not None:
            self.calc_and_update_p_q_sigma(conf.T, conf.flow_type)

        # For model testing properties.
        self.gx = self.gy = self.g
        self.log_counter = 0


    def A(self, t):
        t = torch.tensor(t, device=self.a.device)
        t = t.view(-1, 1)
        t_one_hot = F.one_hot(t, self.conf.T+1).float()
        A = self.a_net(t_one_hot.view(-1, self.conf.T+1))  #.sigmoid()
        # A = A * (t > 0).float()
        # return A + torch.ones_like(A)
        # s = t / self.conf.T
        # c = 2 * (s - 0.5).pow(8)
        # A = c * (1-s) + (1-c) * A
        # return ((A > 0.5).float().detach() + A - A.detach()) / self.a[t].clamp(0.5/self.conf.T, 1.0)
        # return A #/ self.a[t].clamp(0.5/self.conf.T, 1.0)
        # if t.shape[0] == 1:
        #     import pdb; pdb.set_trace()
        return A + torch.cat([torch.ones_like(A[:, :A.shape[1]//2]), torch.zeros_like(A[:, A.shape[1]//2:])], dim=1)
    
    def forward(self, xt, t):
        # Surprisingly, this is almost never used, as training and sampling are done differently.
        # For the final main methods of the class see self.sample and self.train_model.
        gx = self.g(xt)
        g_hat_x0 = self.denoise_g_xt(gx, t)
        return self.g.inverse(g_hat_x0)
    
    def get_losses(self, t, img=None, eps=None):
        if img is None and eps is None:
            raise ValueError("At least one of img or eps must be provided.")
        img = torch.zeros((0, *img.shape[1:]), device=img.device) if img is None else img
        eps = torch.zeros((0, *eps.shape[1:]), device=eps.device) if eps is None else eps
        
        # apply g to both img and eps
        g_out = self.g(torch.cat((img, eps), dim=0))
        g_img, g_eps = torch.split(g_out, (img.shape[0], eps.shape[0]))
        
        loss_img = ((1 - self.a[t][:, None]*self.A(t)) * g_img.view(img.shape[0], -1)).pow(2).mean()
        loss_eps = (self.b[t][:, None]*self.A(t) * g_eps.view(eps.shape[0], -1)).pow(2).mean()
        loss = self.conf.img_w * loss_img + self.conf.eps_w * loss_eps
        
        return loss, loss_img, loss_eps
    
    def sample(self, b_sz=1, nograd=True):
        if self.conf.sample_iterative:
            return self.sample_x0_iterative(b_sz, nograd)
        
        with torch.no_grad() if nograd else torch.enable_grad():
            pass

    def device(self):
        return self.a_net[0].weight.device

    def sample_x0_iterative(self, b_sz=1, nograd=True):
        print(f"Sampling iteratively with T={self.conf.T}")
        with torch.no_grad() if nograd else torch.enable_grad():
            self.eval()
            xT = torch.randn(b_sz, *self.conf.im_shape, device=self.device())
            g_xT = self.g(xT)
            g_xt = g_xT
            for t in range(self.conf.T-1, -1, -1):
                g_hat_x0 = self.denoise_g_xt(g_xt, t+1).view_as(g_xt)
                g_xt = (self.sample_g_xt(g_hat_x0, t, g_xT) + g_xt - 
                        self.sample_g_xt(g_hat_x0, t+1, g_xT))
            hat_x_0 = self.g.inverse(g_xt)
        self.train()
        return hat_x_0

    def denoise_g_xt(self, g_xt, t):
        flat_g_xt = g_xt.view(g_xt.shape[0], -1)
        flat_g_hat_x0 = self.A(t) * flat_g_xt 
        g_hat_x0 = flat_g_hat_x0.view_as(g_xt)
        return g_hat_x0

    def sample_g_xt(self, g_x0, t, g_xT=None):
        with torch.no_grad():
            g_xT = self.g(torch.randn_like(g_x0)) if g_xT is None else g_xT
            return self.a[t] * g_x0 + self.b[t] * g_xT
    
    def train_step(self, img, eps, t):
        loss, loss_img, loss_eps = self.get_losses(t, img, eps)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item(), loss_img.item(), loss_eps.item()
    
    def train_model(self, train_loader, n_epochs):
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
                t = torch.randint(0, self.conf.T+1, (img.shape[0],), device=img.device)
                loss, loss_img, loss_eps = self.train_step(img, eps, t)
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
                                   'Arank': self.A(t).sum().item()}, step=self.log_counter)
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
        grid = make_grid(denorm(generated), nrow=int(sample_bs ** 0.5))
        grid_save_path = os.path.join(self.conf.grid_dir, f"e{epoch}.png")
        imwrite(grid, grid_save_path)
        # we want to make a picture from self.a and save it as well
        t = self.conf.T // 2
        a_img = self.A(t).view(*self.conf.im_shape).repeat(3,1,1)
        a_save_path = os.path.join(self.conf.grid_dir, f"a05_e{epoch}.png")
        imwrite(a_img, a_save_path, bounds=(0,1))
        a_img = self.A(2).view(*self.conf.im_shape).repeat(3,1,1)
        a_save_path = os.path.join(self.conf.grid_dir, f"a0_e{epoch}.png")
        imwrite(a_img, a_save_path, bounds=(0,1))
        a_img = self.A(self.conf.T-2).view(*self.conf.im_shape).repeat(3,1,1)
        a_save_path = os.path.join(self.conf.grid_dir, f"aT_e{epoch}.png")
        imwrite(a_img, a_save_path, bounds=(0,1))
        print(f"[Validation] Generated sample grid saved to {grid_save_path}")
        print(f"images range: {denorm(generated).min().item()} - {denorm(generated).max().item()}")
        print(f"a range: {a_img.min().item()} - {a_img.max().item()}")

        if img is not None:
            img = img[:sample_bs]
            t = self.conf.T // 3
            g_noisy = self.sample_g_xt(self.g(img), t)
            g_denoised = self.denoise_g_xt(g_noisy, t)
            x_denoised = self.g.inverse(g_denoised)
            x_noisy = self.g.inverse(g_noisy)
            denoised_save_path = os.path.join(self.conf.grid_dir, f"denoised_e{epoch}.png")
            imwrite(make_grid(denorm(x_denoised), nrow=int(img.shape[0] ** 0.5)), denoised_save_path, bounds=(0,1))
            print(f"denoised range: {denorm(x_denoised).min().item()} - {denorm(x_denoised).max().item()}")


            noisy_save_path = os.path.join(self.conf.grid_dir, f"noisy_e{epoch}.png")
            imwrite(make_grid(denorm(x_noisy), nrow=int(img.shape[0] ** 0.5)), noisy_save_path, bounds=(0,1))

        if self.conf.save_val_ckpt:
            self.save_checkpoint(f"e{epoch}.pth")
        self.train()

    
    def calc_and_update_p_q_sigma(self, T, flow_type):
        # b = torch.arange(T+1, device=self.device()) / T
        # a = 1 - b

        a_ = torch.arange(T+1, device=self.device()) / T
        a = (1 - a_).sqrt()
        b = a_.sqrt()
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
