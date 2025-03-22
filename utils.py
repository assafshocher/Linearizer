import os
import torch
from datetime import datetime
from data import mnist_denormalize as denorm
from torchvision.utils import make_grid
from PIL import Image
import numpy as np


def imread(fname, bounds=(-1, 1), **kwargs):
    from PIL import Image
    image = Image.open(fname, **kwargs).convert(mode='RGB')
    tensor = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())), dtype=torch.float32)
    tensor = tensor.view(image.size[1], image.size[0], len(image.getbands()))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    vmin, vmax = bounds
    tensor = torch.clamp((vmax - vmin) * tensor + vmin, vmin, vmax)
    return tensor

def imwrite(image, fname, bounds=(0, 1), **kwargs):
    from PIL import Image
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    vmin, vmax = bounds
    # image = (image - vmin) / (vmax - vmin)
    image = (image * 255.0).round().clip(0, 255).to(torch.uint8)
    Image.fromarray(image.permute(1,2,0).cpu().numpy(), 'RGB').save(fname)

def create_gif_from_frames(frames, output_path, duration=100):
    pil_frames = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            pil_frames.append(Image.fromarray(frame))
        elif isinstance(frame, torch.Tensor):
            # Convert torch tensor to numpy array
            if frame.is_cuda:
                frame = frame.cpu()
            if frame.dim() == 4:  # B, C, H, W
                frame = frame.squeeze(0)
            if frame.dim() == 3:  # C, H, W
                frame = frame.permute(1, 2, 0)
            frame_np = frame.numpy()
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame_np))
        else:
            pil_frames.append(frame)
    
    # Save the frames as a GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {output_path}")


def imshow(x, bounds=(0, 1)):
    import matplotlib.pyplot as plt
    vmin, vmax = bounds
    x = x.detach().cpu()
    x = (x - vmin) / (vmax - vmin)
    if x.shape[0] == 1:
        x = x.squeeze(0)
    if x.shape[0] <= 3:
        x = torch.einsum('chw->hwc', x)
    if x.shape[-1] == 1:
        plt.imshow(x.squeeze(-1), cmap='gray')
    else:
        plt.imshow(x)
    plt.show()

def find_latest_checkpoint(folder_path, exp_name=""):
    """
    Returns the latest checkpoint filename (based on modification time)
    in folder_path that contains exp_name in its filename.
    """
    if not os.path.exists(folder_path):
        return None
    files = [f for f in os.listdir(folder_path) if f.endswith(".pth") and exp_name in f]
    if not files:
        return None
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
    return files[-1]


def handle_devices(device):
    if device == "ddp":
        use_ddp = True
        import torch.distributed as dist
        # DDP: assume launch with torchrun or similar so that LOCAL_RANK is set.
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        try:
            device = torch.device(device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise ValueError("CUDA is not available but a CUDA device was specified.")
        except Exception as e:
            raise ValueError(f"Invalid device specified: {device}") from e
        world_size = 1
        rank = 0
        use_ddp = False
    return use_ddp, device, world_size, rank



def create_experiment_dirs(results_dir, dataset, exp_name=""):

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dataset}_{date_str}"
    if exp_name:
        base_name += f"_{exp_name}"
    exp_dir = os.path.join(results_dir, base_name)
    ckpt_dir = os.path.join(exp_dir, "ckpts")
    grid_dir = os.path.join(exp_dir, "generated")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)
    return exp_dir, ckpt_dir, grid_dir


def add_singletons_as(x, y, shift=0, start_dim=0):
    return x.view(x.shape[:start_dim] + 
                (1,) * (y.dim() + shift - x.dim()) + 
                x.shape[start_dim:])


def save_visuals(results, frames, g_inv, dir, epoch=None):
    # Create a dict of grids for all the results
    grids = {}
    for key, value in results.items():
        grids[key] = make_grid(denorm(value), nrow=int(value.shape[0] ** 0.5))
    
    border_width = 5  # Thickness of white line
    device = frames[0][0].device  # Get the device of the grid tensor
    border_h = torch.ones(3, grids["g_xt"].shape[1], 
                          border_width, device=device)  # Horizontal white bar
    
    # Concatenate with white borders in between for xt
    concat_xt = torch.cat([
        grids["xt"], border_h, 
        grids["g_xt"], border_h, 
        grids["A_g_xt"], border_h,
        grids["denoised_xt"]], dim=2)  
    
    concat_xT = torch.cat([
        grids["xT"], border_h, 
        grids["g_xT"], border_h, 
        grids["A_g_xT"], border_h,
        grids["denoised_xT"]], dim=2)
    
    concat_x0 = torch.cat([
        grids["x0"], border_h, 
        grids["g_x0"], border_h, 
        grids["A_g_x0"], border_h,
        grids["denoised_x0"]], dim=2)
    
    # Save concatenated visualizations
    imwrite(concat_xt, os.path.join(dir, f"xt_e{epoch}.png"))
    imwrite(concat_xT, os.path.join(dir, f"xT_e{epoch}.png"))
    imwrite(concat_x0, os.path.join(dir, f"x0_e{epoch}.png"))

    # Create and save GIFs of the sampling process
    grid_frames_xt = []
    grid_frames_hat_x0 = []
    grid_frames_concat = []
        
    # Process frames
    for frame in frames:
        with torch.no_grad():
            g_xt, g_hat_x0, t = frame
            xt = g_inv(g_xt)
            x0 = g_inv(g_hat_x0)
        
        # Create individual grids for each component
        grid_xt = make_grid(denorm(xt), nrow=int(xt.shape[0] ** 0.5))
        grid_hat_x0 = make_grid(denorm(x0), nrow=int(x0.shape[0] ** 0.5))
        grid_g_xt = make_grid(denorm(g_xt), nrow=int(g_xt.shape[0] ** 0.5))
        grid_g_hat_x0 = make_grid(denorm(g_hat_x0), nrow=int(g_hat_x0.shape[0] ** 0.5))
  
        # Store individual grids for simple GIFs
        grid_frames_xt.append(grid_xt)
        grid_frames_hat_x0.append(grid_hat_x0)
        
        border_width = 5  # Thickness of white line
        device = grid_xt.device  # Get the device of the grid tensor
        border_h = torch.ones(3, grid_xt.shape[1], border_width, device=device)  # Horizontal white bar
        
        # Create concatenated visualization with white borders
        concat_grid = torch.cat([
            grid_xt, border_h,
            grid_g_xt, border_h,
            grid_g_hat_x0, border_h,
            grid_hat_x0], dim=2) 
        
        grid_frames_concat.append(concat_grid)
    
    # Create and save GIFs
    epoch_suffix = f"_e{epoch}" if epoch is not None else ""
  
    # Save the concatenated visualization GIF
    concat_gif_path = os.path.join(dir, f"sampling_process{epoch_suffix}.gif")
    create_gif_from_frames(grid_frames_concat, concat_gif_path, duration=100)
    

