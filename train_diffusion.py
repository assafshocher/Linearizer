import os
import argparse
from datetime import datetime
import torch
from utils import create_experiment_dirs, handle_devices
from lin_diff_2 import LinearDiffusion
from data import get_data_loaders


def main():
    parser = argparse.ArgumentParser()
    # General training arguments.
    parser.add_argument("--device", type=str, required=True,
                        help="Device to use, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--val_batch_size", type=int, default=256, help="Validation batch size.")
    parser.add_argument("--log_freq", type=int, default=10, help="Log frequency (in steps).")
    parser.add_argument("--val_freq", type=int, default=1, help="Validation frequency (in epochs).")
    parser.add_argument("--save_val_ckpt", action="store_true", default=False,
                        help="Save a checkpoint after validation.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 penalty).")
    parser.add_argument("--use_ddp", action="store_true", default=False,
                        help="Use DistributedDataParallel (DDP).")
    
    # Experiment & folder settings.
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Base directory to save experiment results.")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Additional text to add to the experiment folder name.")
    
    # Dataset arguments.
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset to use. Options: mnist, cifar10, cifar100, imagenet, celeba")
    parser.add_argument("--orig_im_size", type=int, default=28,
                        help="Original image size (used for augmentations). For MNIST, default=28.")
    parser.add_argument("--target_im_size", type=int, default=64,
                        help="Target image size (used for training). For MNIST, default=32.")
    
    # Model-specific arguments.
    parser.add_argument("--n_layers", type=int, default=5, help="Number of layers in g.")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads in g.")
    parser.add_argument("--p_sz", type=int, default=8, help="Patch size for g.")
    parser.add_argument("--im_shape", type=int, nargs=3, default=[1, 64, 64],
                        help="Image shape as C H W for the model (target size).")
    
    # Diffusion process arguments.
    parser.add_argument("--T", type=int, default=1000, help="Number of timesteps for the diffusion process.")
    parser.add_argument("--flow_type", type=str, default="DDPM", help="Flow type (e.g., DDPM, DDIM).")
    parser.add_argument("--betas", type=float, nargs='*', default=None, help="Optional beta schedule (list of floats).")
    parser.add_argument("--beta_min", type=float, default=1e-4, help="Minimum beta for default schedule.")
    parser.add_argument("--beta_max", type=float, default=0.02, help="Maximum beta for default schedule.")
    parser.add_argument("--img_w", type=float, default=1.0, help="Weight for image loss.")
    parser.add_argument("--eps_w", type=float, default=1.0, help="Weight for epsilon loss.")
    parser.add_argument("--sample_iterative", action="store_true", default=False,
                        help="Sample iteratively instead of one-step.")
    parser.add_argument("--wandb", type=bool, default=False)

    # Checkpoint loading flag.
    parser.add_argument("--load_latest", action="store_true", default=False,
                        help="If set, load the latest checkpoint for this experiment.")

    conf = parser.parse_args()
    print(conf)
    if conf.wandb:
        import wandb
        wandb.init(config=conf, project="Linearizer")
    # --- Device & DDP handling ---
    conf.use_ddp, conf.device, world_size, rank = handle_devices(conf.device)
    
    # --- Experiment folder handling ---
    # Create an experiment folder with dataset, current date, and exp_name.
    exp_dir, ckpt_dir, grid_dir = create_experiment_dirs(conf.results_dir, conf.dataset, conf.exp_name)
    conf.ckpt_dir = ckpt_dir
    conf.grid_dir = grid_dir
    print(f"Experiment directory: {exp_dir}")
    
    # --- Data loaders ---
    train_loader, val_loader = get_data_loaders(
                               conf.dataset,
                               conf.batch_size,
                               conf.val_batch_size,
                               conf.orig_im_size,
                               conf.target_im_size,
                               use_ddp=conf.use_ddp,
                               world_size=world_size,
                               rank=rank)
    

    # --- Initialize the model ---
    model = LinearDiffusion(conf).to(conf.device)
    
    # If requested, load the latest checkpoint.
    if conf.load_latest:
        model.load_checkpoint()

    # --- Start training ---
    model.train_model(train_loader, conf.n_epochs, val_loader=val_loader)

    # Save the final model checkpoint.
    final_ckpt_path = os.path.join(conf.ckpt_dir, f"linear_diffusion_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Training complete. Model saved as '{final_ckpt_path}'.")

    if conf.use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
