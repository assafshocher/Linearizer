import torch
import matplotlib.pyplot as plt


def plot_transitions(linearizer, x_image):
    """Plot transitions from x to y's with each matrix"""
    linearizer.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        style_names = ['mosaic', 'candy', 'rain', 'udine']

        axes[0].imshow(x_image[0].cpu().permute(1, 2, 0))
        axes[0].set_title('Input X')
        axes[0].axis('off')

        for i in range(4):
            y_pred = linearizer.gy_inverse(linearizer.A(linearizer.gx(x_image), t=torch.tensor([i]).to(x_image.device)))
            axes[i + 1].imshow(y_pred[0].cpu().permute(1, 2, 0))
            axes[i + 1].set_title(f'Predicted Y_{style_names[i]}')
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.savefig('style_transitions.png')
        plt.show()


def plot_data(x_image, y_images):
    """Display all x and y images side by side"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    style_names = ['mosaic', 'candy', 'rain', 'udine']

    axes[0].imshow(x_image[0].cpu().permute(1, 2, 0))
    axes[0].set_title('X_input')
    axes[0].axis('off')

    for i in range(4):
        axes[i + 1].imshow(y_images[i].cpu().permute(1, 2, 0))
        axes[i + 1].set_title(f'Y_{style_names[i]}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig('byb_dataset_visualization.png')
    plt.show()
