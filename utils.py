import os
import torch
from datetime import datetime

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
    """
    Creates an experiment folder inside results_dir with a name based on the dataset,
    current date, and optional exp_name. Returns (exp_dir, ckpt_dir, grid_dir).
    """
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



