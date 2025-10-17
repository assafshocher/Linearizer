import time
import json
import torch
import os
import sys
from datetime import datetime

sys.path.append('..')
sys.path.append(os.getcwd())

from one_step.data.data_utils import get_data_loaders
from one_step.modules.one_step_linearizer import OneStepLinearizer
from one_step.utils.loss_utils import calculate_lpips
from one_step.utils.model_utils import get_linear_network, get_g
from one_step.utils.sampling_utils import sample_and_save
from configs.celeba import get_celeba_parser
from configs.mnist import get_mnist_parser


def parse_args():
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        if dataset == 'celeba':
            parser = get_celeba_parser()
        elif dataset == 'mnist':
            parser = get_mnist_parser()
        else:
            raise NotImplementedError(f'Dataset {dataset} not implemented')
        # Remove dataset from args to avoid conflict
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    else:
        raise NotImplementedError(f'No argument! insert dataset name as first argument [mnist, celeba],'
                                  f' and then the rest of the arguments')

    return parser.parse_args()


class FlowMatcher:
    def __init__(self, linearizer: OneStepLinearizer):
        self.linearizer = linearizer

    def training_losses(self, x1, x0=None, noise_level=0.0):
        batch_size = x1.shape[0]
        device = x1.device

        if x0 is None:
            x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=device)

        # --- project into induced space --- #
        g_x0 = self.linearizer.gy(x0)
        g_x1 = self.linearizer.gx(x1)

        # --- predict in the induced space --- #
        g_xt = (1.0 - t)[:, None, None, None] * g_x0 + t[:, None, None, None] * g_x1
        g_x1_p = self.linearizer.A(g_xt, t=t)

        # --- calculate losses --- #
        induced_space_loss = ((g_x1_p - g_x1) ** 2).mean()

        # add regularizing noise
        g_x0 = g_x0 + torch.randn_like(g_x1) * noise_level
        g_x1 = g_x1 + torch.randn_like(g_x1) * noise_level
        # calculate reconstruction losses
        x0_rec_loss = ((x0 - self.linearizer.gy_inverse(g_x0)) ** 2).mean()
        x1_rec_loss = calculate_lpips(x1, self.linearizer.gx_inverse(g_x1))
        x1_pred_rec_loss = calculate_lpips(x1, self.linearizer.gx_inverse(g_x1_p))
        # final loss
        loss = induced_space_loss + x0_rec_loss + x1_rec_loss + x1_pred_rec_loss

        return loss

    def sample(self, x, device, steps=100, method='euler', return_path=False):
        self.linearizer.eval()

        with torch.no_grad():
            g_x = self.linearizer.gy(x)  # g_x strat as g_y (the encoding of the noise)
            dt = 1.0 / steps

            if return_path:
                path = [g_x]

            if method == 'euler':
                for i in range(0, steps - 1):
                    t = torch.full((g_x.shape[0],), i * dt, device=device)
                    g_t_model = self.linearizer.A(g_x, t=t)
                    g_vt = (g_t_model - g_x) / (1 - t)[:, None, None, None]
                    g_x = g_x + g_vt * dt
                    if return_path:
                        path.append(g_x)

            elif method == 'rk':
                for i in range(0, steps - 1):
                    t = torch.full((g_x.shape[0],), i * dt, device=device)

                    # k1
                    g_t_model = self.linearizer.A(g_x, t=t)
                    k1 = (g_t_model - g_x) / (1 - t)[:, None, None, None]

                    # k2
                    g_x_k2 = g_x + 0.5 * dt * k1
                    t_k2 = t + 0.5 * dt
                    g_t_model_k2 = self.linearizer.A(g_x_k2, t=t_k2)
                    k2 = (g_t_model_k2 - g_x_k2) / (1 - t_k2)[:, None, None, None]

                    # k3
                    g_x_k3 = g_x + 0.5 * dt * k2
                    g_t_model_k3 = self.linearizer.A(g_x_k3, t=t_k2)
                    k3 = (g_t_model_k3 - g_x_k3) / (1 - t_k2)[:, None, None, None]

                    # k4
                    g_x_k4 = g_x + dt * k3
                    t_k4 = t + dt
                    g_t_model_k4 = self.linearizer.A(g_x_k4, t=t_k4)
                    k4 = (g_t_model_k4 - g_x_k4) / (1 - t_k4)[:, None, None, None]

                    # Update
                    g_x = g_x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

                    if return_path:
                        path.append(g_x)

            g_x = self.linearizer.gx_inverse(g_x)

        if return_path:
            return g_x, path

        return g_x

    def sample_one_step(self, x, device, sampling_method='rk', T=100, B=None):
        self.linearizer.eval()
        with torch.no_grad():
            g_x = self.linearizer.gy(x)  # g_x strat as g_y (the encoding of the noise)
            if B is None:
                B = self.get_sampling_terms(device, sampling_method=sampling_method, T=T)
            B = B.to(device)
            g_x = (g_x.reshape(g_x.shape[0], -1) @ B).reshape(g_x.shape)
            g_x = self.linearizer.gx_inverse(g_x)
        return g_x

    def get_sampling_terms(self, device, T=100, sampling_method='euler'):
        with torch.no_grad():
            I = torch.eye(self.linearizer.net_gy.dim).to(device)
            B = I
            dt = 1.0 / T

            if sampling_method == 'euler':
                for i in range(T - 2, -1, -1):
                    t_k = i * dt
                    A_t_k = self.linearizer.linear_network.get_lin_t(torch.ones(1).to(device) * t_k).squeeze(0)
                    M_k = I + (dt / (1.0 - t_k)) * (A_t_k - I)
                    B = M_k @ B

            elif sampling_method == 'rk':
                for i in range(T - 2, -1, -1):
                    t_k = i * dt
                    A_t_k = self.linearizer.linear_network.get_lin_t(torch.ones(1).to(device) * t_k).squeeze(0)
                    k1 = (dt / (1.0 - t_k)) * (A_t_k - I)

                    t_k2 = t_k + 0.5 * dt
                    A_t_k2 = self.linearizer.linear_network.get_lin_t(torch.ones(1).to(device) * t_k2).squeeze(0)
                    k2 = (dt / (1.0 - t_k2)) * (A_t_k2 - I)
                    k3 = (dt / (1.0 - t_k2)) * (A_t_k2 - I)

                    t_k4 = t_k + dt
                    A_t_k4 = self.linearizer.linear_network.get_lin_t(torch.ones(1).to(device) * t_k4).squeeze(0)
                    k4 = (dt / (1.0 - t_k4)) * (A_t_k4 - I)

                    M_k = I + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    B = M_k @ B
            return B


def train_flow_matching(linearizer, dataloader, epochs=10, lr=1e-4, noise_level=0.0, eval_epoch=10, steps=100,
                        num_of_ch=1, sampling_method='rk', save_folder='', img_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init flow matching framework with linearizer
    linearizer = linearizer.to(device)
    fm = FlowMatcher(linearizer)
    optimizer = torch.optim.Adam([{"params": linearizer.parameters(), "lr": lr}, ], betas=(0.9, 0.999),
                                 weight_decay=0.0)

    # save paths
    models_save_path = f'{save_folder}/models'
    artifacts_save_path = f'{save_folder}/artifacts'
    os.makedirs(models_save_path, exist_ok=True), os.makedirs(artifacts_save_path, exist_ok=True)

    # training
    for epoch in range(epochs):
        total_loss = 0
        linearizer.train()
        for batch_idx, (x1, lbl) in enumerate(dataloader):
            x1 = x1.to(device)
            optimizer.zero_grad()
            loss = fm.training_losses(x1=x1, x0=None, noise_level=noise_level)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs} completed, Avg Loss: {avg_loss:.4f}')

        if epoch % eval_epoch == 0:
            linearizer.eval()
            print("Generating samples...")
            sample_and_save(fm=fm, num_of_images=16, device=device, steps=steps,
                            epoch=epoch, num_of_ch=num_of_ch, sampling_method=sampling_method,
                            img_size=img_size, save_dir=artifacts_save_path)
            # Save model

            torch.save(
                fm.linearizer.state_dict(),
                f'{models_save_path}/{epoch}.pth')
            print(f"Model saved to {models_save_path}/{epoch}.pth")


def main():
    # --- parse and prepare arguments ---
    args = parse_args()
    print(f"arguments: {args}")

    # --- load data --- #
    dataloader, _ = get_data_loaders(args.dataset,
                                     args.batch_size,
                                     args.batch_size_val,
                                     target_size=args.img_size)

    print(f"loaded dataset {args.dataset}")

    # --- create models --- #
    linear_network = get_linear_network(args.linear_module,
                                        linear_lora_features=args.linear_lora_features,
                                        in_ch=args.in_ch,
                                        img_size=args.img_size)

    g = get_g(args.g, args.in_ch, args.out_ch, args.img_size)
    linearizer = OneStepLinearizer(gx=g, linear_network=linear_network)

    # --- start training --- #
    print("Starting Flow Matching training...")
    save_folder = f'{args.save_folder}/{args.dataset}/{datetime.now().strftime("%m_%d_%H_%M_%S")}'
    os.makedirs(save_folder, exist_ok=True)

    # Save args
    with open(f'{save_folder}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    train_flow_matching(
        linearizer=linearizer,
        dataloader=dataloader,
        epochs=args.epc,
        noise_level=args.noise_level,
        steps=args.steps,
        sampling_method=args.sampling_method,
        save_folder=save_folder,
        img_size=args.img_size,
        num_of_ch=args.in_ch,
    )
    print("Training completed!")


if __name__ == '__main__':
    main()
