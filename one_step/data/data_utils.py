from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np


def get_mnist_data_loaders(train_bs, val_bs, target_size, use_ddp=False, world_size=1, rank=0):
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def get_celeba_data_loaders(train_bs, val_bs, target_size, use_ddp=False, world_size=1, rank=0):
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CelebA(root='./data', split='train', download=True, transform=train_transform)
    val_dataset = datasets.CelebA(root='./data', split='valid', download=True, transform=test_transform)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


def get_data_loaders(dataset_name, train_bs, val_bs, target_size=32, use_ddp=False, world_size=1, rank=0):
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return get_mnist_data_loaders(train_bs, val_bs, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'celeba':
        return get_celeba_data_loaders(train_bs, val_bs, target_size, use_ddp, world_size, rank)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
