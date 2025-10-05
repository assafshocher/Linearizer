import torch
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
import torch.nn.functional as F
from piq import LPIPS

sys.path.append(os.getcwd())

from linearizer.one_step.data.data_utils import get_data_loaders
from linearizer.one_step.modules.one_step_linearizer import OneStepLinearizer
from linearizer.one_step.utils.model_utils import get_linear_network, get_g
from linearizer.one_step.train_one_step import FlowMatcher


def load_model(model_path, device='cuda'):
    """Load trained linearizer model using saved args"""
    # Load args from the same directory
    model_dir = os.path.dirname(os.path.dirname(model_path))  # Go up to run folder
    args_path = os.path.join(model_dir, 'args.json')

    with open(args_path, 'r') as f:
        args = json.load(f)

    # Reconstruct the linearizer architecture using saved args
    linear_network = get_linear_network(args['linear_module'],
                                        linear_lora_features=args['linear_lora_features'],
                                        in_ch=args['in_ch'],
                                        img_size=args['img_size'])

    g = get_g(args['g'], args['in_ch'], args['out_ch'], args['img_size'])
    linearizer = OneStepLinearizer(gx=g, linear_network=linear_network)

    # Load state dict
    linearizer.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    linearizer = linearizer.to(device)

    # Create FlowMatcher
    fm = FlowMatcher(linearizer)
    return fm, args


def generate_samples(fm, num_ch, img_size, num_samples=24, device='cuda', steps=100, method='rk',
                     seed_list=None, one_step=False):
    """Generate samples using one-step or multi-step"""
    if seed_list is None:
        seed_list = list(range(num_samples))

    x = torch.zeros(num_samples, num_ch, img_size, img_size, device=device)
    for i in range(num_samples):
        torch.manual_seed(seed_list[i])
        x[i] = torch.randn(1, num_ch, img_size, img_size, device=device)

    if one_step:
        samples = fm.sample_one_step(x, device, sampling_method=method, T=steps)
    else:
        samples = fm.sample(x, device, steps=steps, method=method)

    return samples


def save_samples_grid(samples, output_path='generated_samples.pdf', dataset='mnist'):
    """Save samples in 4x6 grid"""
    cmap = 'gray' if dataset == 'mnist' else None
    samples = torch.clamp(samples, 0, 1)

    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            if dataset == 'mnist':
                img = samples[i].cpu().squeeze().numpy()
            else:
                img = samples[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img, cmap=cmap)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def interpolation_experiment(fm, dataloader, device='cuda', num_interpolation_steps=10, sample_pairs=None):
    """Perform interpolation experiments"""
    if sample_pairs is None:
        sample_pairs = [[(1, 3), (8, 10), (15, 17), (18, 20)], [(32, 34), (35, 37), (34, 36), (51, 53)]]

    # Get batch from dataloader
    x1_batch, _ = next(iter(dataloader))
    x1_batch = x1_batch.to(device)

    lpips_fn = LPIPS(replace_pooling=True, reduction="mean").to(device)

    for pairs in sample_pairs:
        all_psnr_scores = []
        all_lpips_scores = []

        with torch.no_grad():
            # Get sampling terms
            B = fm.get_sampling_terms(device, sampling_method='rk', T=6)
            B_inv = torch.pinverse(B)

            all_interpolations = []

            for a, b in pairs:
                x1_1, x1_2 = x1_batch[a:a + 1], x1_batch[b:b + 1]

                # Project to latent space
                g_x1_1 = fm.linearizer.gx(x1_1)
                g_x1_2 = fm.linearizer.gx(x1_2)

                # Project to noise space
                g_x0_1 = (g_x1_1.reshape(-1) @ B_inv).reshape(g_x1_1.shape)
                g_x0_2 = (g_x1_2.reshape(-1) @ B_inv).reshape(g_x1_2.shape)

                # Project to x0 space
                x0_1 = fm.linearizer.gy_inverse(g_x0_1)
                x0_2 = fm.linearizer.gy_inverse(g_x0_2)

                # Interpolate in x0 space
                x0_interp = []
                for i in range(num_interpolation_steps):
                    alpha = i / (num_interpolation_steps - 1)
                    x0_i = (1 - alpha) * x0_1 + alpha * x0_2
                    x0_interp.append(x0_i)

                # Project back to x1 space
                x1_interp = []
                for x0_i in x0_interp:
                    g_x0_i = fm.linearizer.gy(x0_i)
                    g_x1_i = (g_x0_i.reshape(-1) @ B).reshape(g_x0_i.shape)
                    x1_i = fm.linearizer.gx_inverse(g_x1_i)
                    x1_interp.append(x1_i)

                all_interpolations.append((x1_1, x1_interp, x1_2))

                # Calculate metrics
                x1_1_rec = x1_interp[0]

                psnr_1 = -10 * torch.log10(F.mse_loss(x1_1, x1_1_rec))
                lpips_1 = lpips_fn(x1_1, x1_1_rec)

                all_psnr_scores.append(psnr_1.item())
                all_lpips_scores.append(lpips_1.item())

        # Plot interpolations
        fig, axes = plt.subplots(num_interpolation_steps + 2, 4, figsize=(8, (num_interpolation_steps + 2) * 2))
        cmap = 'gray' if 'mnist' in str(fm).lower() else None

        for col, (x1_1, x1_interp, x1_2) in enumerate(all_interpolations):
            # Original images
            img1 = torch.clamp(x1_1, 0, 1).squeeze().cpu()
            if len(img1.shape) == 3:
                img1 = img1.permute(1, 2, 0)
            axes[0, col].imshow(img1.numpy(), cmap=cmap)
            axes[0, col].axis('off')

            # Interpolations
            for i, x1_i in enumerate(x1_interp):
                img = torch.clamp(x1_i, 0, 1).squeeze().cpu()
                if len(img.shape) == 3:
                    img = img.permute(1, 2, 0)
                axes[i + 1, col].imshow(img.numpy(), cmap=cmap)
                axes[i + 1, col].axis('off')

            # Target image
            img2 = torch.clamp(x1_2, 0, 1).squeeze().cpu()
            if len(img2.shape) == 3:
                img2 = img2.permute(1, 2, 0)
            axes[num_interpolation_steps + 1, col].imshow(img2.numpy(), cmap=cmap)
            axes[num_interpolation_steps + 1, col].axis('off')

        plt.tight_layout()
        output_path = f'interpolation_result_{pairs}.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print metrics
        avg_psnr = sum(all_psnr_scores) / len(all_psnr_scores)
        avg_lpips = sum(all_lpips_scores) / len(all_lpips_scores)
        print(f"Interpolation PSNR: {avg_psnr:.4f} dB")
        print(f"Interpolation LPIPS: {avg_lpips:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Test one-step generation model')
    parser.add_argument('--model_path', type=str, required=True, help='path/to/model/model.pth')
    return parser.parse_args()


def main():
    test_args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and args
    fm, args = load_model(test_args.model_path, device)
    dataset = args['dataset']
    img_size = args['img_size']

    # Load data for interpolation
    dataloader, _ = get_data_loaders(dataset, 64, 64, target_size=img_size)

    print("Generating samples...")

    # Generate samples
    grid_seeds = list(range(24))
    samples_one_step = generate_samples(fm, args['in_ch'], args['img_size'], num_samples=24, device=device, steps=100,
                                        seed_list=grid_seeds, one_step=True)
    samples_multi_step = generate_samples(fm, args['in_ch'], args['img_size'], num_samples=24, device=device, steps=100,
                                          seed_list=grid_seeds, one_step=False)

    # Save samples
    save_samples_grid(samples_one_step, f'generated_samples_{dataset}_one_step.pdf', dataset)
    save_samples_grid(samples_multi_step, f'generated_samples_{dataset}_multi_step.pdf', dataset)

    # Calculate comparison metrics
    lpips_fn = LPIPS(replace_pooling=True, reduction="mean").to(device)
    with torch.no_grad():
        psnr_scores = []
        lpips_scores = []

        for i in range(24):
            psnr = -10 * torch.log10(F.mse_loss(samples_one_step[i:i + 1], samples_multi_step[i:i + 1]))
            lpips = lpips_fn(samples_one_step[i:i + 1], samples_multi_step[i:i + 1])
            psnr_scores.append(psnr.item())
            lpips_scores.append(lpips.item())

        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        avg_lpips = sum(lpips_scores) / len(lpips_scores)

    print(f"One-step vs Multi-step PSNR: {avg_psnr:.4f} dB")
    print(f"One-step vs Multi-step LPIPS: {avg_lpips:.4f}")

    # Interpolation experiments
    print("Performing interpolation experiments...")
    sample_pairs = [[(1, 3), (8, 10), (15, 17), (18, 20)]] if dataset == 'mnist' else [
        [(22, 24), (14, 16), (13, 15), (26, 28)]]
    interpolation_experiment(fm, dataloader, device, num_interpolation_steps=10, sample_pairs=sample_pairs)

    print("Inference completed!")


if __name__ == '__main__':
    main()
