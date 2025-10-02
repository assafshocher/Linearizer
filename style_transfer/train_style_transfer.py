import os
import torch.optim as optim
import torch
import argparse
from datetime import datetime

from linearizer.common.song__unet import creat_song_unet
from linearizer.style_transfer.modules.invertable_network import InverseUnet
from linearizer.style_transfer.modules.linear_network import LinearKernel
from linearizer.style_transfer.modules.style_linearizer import StyleLinearizer
from linearizer.style_transfer.utils import plot_data, plot_transitions


def load_data(path_to_images):
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    y_images = []
    style_names = ['mosaic', 'candy', 'rain_princess', 'udnie']
    for style in style_names:
        img = Image.open(f'{path_to_images}/{style}.png').convert('RGB')
        y_images.append(transform(img))

    x_img = Image.open(f'{path_to_images}/original.png').convert('RGB')
    x_tensor = transform(x_img)

    return x_tensor, torch.stack(y_images)


def parse_args():
    parser = argparse.ArgumentParser(description='Train style transfer linearizer')
    parser.add_argument('--path_to_images', type=str, default='celeb', help='Path to images directory')
    parser.add_argument('--img_resolution', type=int, default=256, help='Image resolution')
    return parser.parse_args()


def train_linearizer():
    args = parse_args()
    path_to_images = args.path_to_images
    img_resolution = args.img_resolution
    output_path = f'outputs/{datetime.now().strftime("%m_%d_%H_%M_%S")}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_image, y_images = load_data(path_to_images)

    x_image = x_image.to(device).unsqueeze(0)  # Add batch dimension
    y_images = y_images.to(device)
    x_repeat = x_image.repeat(4, 1, 1, 1)

    plot_data(x_image, y_images)

    # Create linearizer model
    gx = InverseUnet(3, 3, img_resolution, creat_song_unet)
    A = LinearKernel()
    linearizer = StyleLinearizer(gx=gx, linear_network=A).to(device)
    optimizer = optim.Adam(linearizer.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    t = torch.tensor(list(range(4))).to(device)
    epochs = 2000

    # Training loop
    linearizer.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = linearizer.gy_inverse(linearizer.A(linearizer.gx(x_repeat), t=t))
        loss = criterion(y_pred, y_images)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}')

    plot_transitions(linearizer, x_image)
    os.makedirs(f'{output_path}', exist_ok=True)
    torch.save(linearizer, f'{output_path}/model.pth')

    print(f"Linearizer model saved as {output_path}/model.pth")


if __name__ == "__main__":
    train_linearizer()
