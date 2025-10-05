from PIL import Image
import os
import argparse
import zipfile
import subprocess
import sys

try:
    from torch.utils.model_zoo import _download_url_to_file
except ImportError:
    try:
        from torch.hub import download_url_to_file as _download_url_to_file
    except ImportError:
        from torch.hub import _download_url_to_file


def parse_args():
    parser = argparse.ArgumentParser(description='Apply neural style transfer to image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--height', type=int, default=256, help='Target height')
    parser.add_argument('--width', type=int, default=256, help='Target width')
    return parser.parse_args()


def download_models():
    """Download style transfer models if not already present"""
    if os.path.exists('saved_models'):
        print("Style models already exist, skipping download")
        return

    print("Downloading style transfer models...")
    _download_url_to_file('https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=1', 'saved_models.zip', None,
                          True)

    with zipfile.ZipFile('saved_models.zip') as zf:
        zf.extractall('.')

    os.remove('saved_models.zip')
    print("Models downloaded successfully")


def apply_style_transfer(content_image, style_name, model_path, output_path):
    """Apply style transfer using neural_style"""
    cmd = [
        sys.executable, 'neural_style/neural_style.py', 'eval',
        '--content-image', content_image,
        '--model', model_path,
        '--output-image', output_path,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Style '{style_name}' applied successfully: {output_path}")
    except subprocess.CalledProcessError:
        print(f"Failed to apply style '{style_name}'")


def main():
    args = parse_args()

    # Resize input image
    img = Image.open(args.image_path)
    resized_img = img.resize((args.width, args.height))

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    resized_path = f"{base_name}/original.png"
    os.makedirs(base_name, exist_ok=True)
    resized_img.save(resized_path)
    print(f"Resized image saved as: {resized_path}")

    # Download models
    download_models()

    # Define 4 different styles
    styles = [
        ('candy', 'saved_models/candy.pth'),
        ('udnie', 'saved_models/udnie.pth'),
        ('rain_princess', 'saved_models/rain_princess.pth'),
        ('mosaic', 'saved_models/mosaic.pth')
    ]

    # Apply each style
    for style_name, model_path in styles:
        if os.path.exists(model_path):
            output_path = f"{base_name}/{style_name}.png"
            apply_style_transfer(resized_path, style_name, model_path, output_path)
        else:
            print(f"Model not found: {model_path}")

    print("Style transfer completed!")


if __name__ == '__main__':
    main()
