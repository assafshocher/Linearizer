import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

# Statistics for CIFAR-100
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)

def get_data_loaders(batch_size_t,
                     batch_size_v=2048,
                     use_ddp=False,
                     world_size=1,
                     rank=0):
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD),])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD),])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_t,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_v,
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True
        )

    else:
        # Normal single-GPU or DataParallel
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_t,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_v,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    return train_loader, test_loader
