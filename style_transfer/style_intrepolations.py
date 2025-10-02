import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Style transfer interpolation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--output_dir', type=str, default='interpolation_results', help='Output directory')
    return parser.parse_args()


def load_x_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    x_img = Image.open(image_path).convert('RGB')
    return transform(x_img).unsqueeze(0)


def interpolate_styles(linearizer, style1, style2, x_image, steps=9):
    device = next(linearizer.parameters()).device
    x_image = x_image.to(device)

    # Linear interpolation between style indices
    alphas = torch.tensor([0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 1])
    interpolated_images = []

    with torch.no_grad():
        for alpha in alphas:
            # Interpolate style parameter
            t_interp = (1 - alpha) * style1 + alpha * style2
            t_tensor = torch.tensor([t_interp]).to(device)

            # Generate interpolated image
            y_pred = linearizer.gy_inverse(linearizer.A(linearizer.gx(x_image), t=t_tensor))
            interpolated_images.append(y_pred[0].cpu().clamp(0, 1))

    return interpolated_images


def plot_interpolation(images, style1, style2, output_dir, style_names):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))
    if len(images) == 1:
        axes = [axes]

    for i, img in enumerate(images):
        axes[i].imshow(img.permute(1, 2, 0).clamp(0, 1))
        axes[i].axis('off')

    plt.tight_layout()
    save_name = f'interpolation_{style_names[style1]}_{style_names[style2]}.pdf'
    plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_name}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    linearizer = torch.load(args.model_path, map_location=device, weights_only=False)
    linearizer.eval()

    # Load input image
    x_image = load_x_image(args.image_path)

    # Style pairs to interpolate
    style_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    style_names = ['mosaic', 'candy', 'rain_princess', 'udnie']

    print(f"Generating interpolations for {len(style_pairs)} style pairs...")

    for style1, style2 in style_pairs:
        print(f"Interpolating between {style_names[style1]} and {style_names[style2]}")

        # Generate interpolated images
        images = interpolate_styles(linearizer, style1, style2, x_image, steps=9)

        # Plot results
        plot_interpolation(images, style1, style2, args.output_dir, style_names)

    print(f"All interpolations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
