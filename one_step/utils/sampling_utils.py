import torch
import matplotlib.pyplot as plt


def sample_and_save(fm, num_of_images, device, epoch, save_dir, num_of_ch, steps=100, sampling_method='rk',
                    img_size=32):
    x = torch.randn((num_of_images, num_of_ch, img_size, img_size), device=device)

    # One step samples
    samples_one_step = fm.sample_one_step(x, device, sampling_method=sampling_method, T=steps)
    save_one_step_sample(num_of_images, save_dir, f"one_{epoch}", samples_one_step)

    # Multi step samples
    samples = fm.sample(x, device, steps=steps, method=sampling_method)
    save_one_step_sample(num_of_images, save_dir, f"multi_{epoch}", samples)


def save_one_step_sample(k, path, name, samples_one_step):
    """Save generated samples as images"""
    p = 'cifar' in path or 'celeb' in path
    samples_one_step = torch.clamp(samples_one_step, 0, 1)
    cmap = None if p else 'gray'

    grid_size = int(k ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < k:
            img = samples_one_step[i, :].cpu().permute(1, 2, 0).squeeze().numpy() if p else samples_one_step[
                i, 0].cpu().numpy()
            ax.imshow(img, cmap=cmap)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{path}/{name}.png', dpi=150, bbox_inches='tight')
    print(f'Generated {k} samples saved to {path}/{name}.png')
