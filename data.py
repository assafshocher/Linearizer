from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_mnist_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # For MNIST: original size is typically 28; target size may differ.
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def mnist_denormalize(x):
    return x * 0.3081 + 0.1307


def get_cifar10_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # CIFAR-10 normalization values.
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomCrop(target_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def get_cifar100_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # CIFAR-100 normalization statistics.
    mean = (0.5071, 0.4865, 0.4409)
    std  = (0.2673, 0.2564, 0.2762)
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomCrop(target_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def get_imagenet_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # Typical ImageNet normalization and transforms.
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize(int(target_size * 1.14)),
        transforms.RandomResizedCrop(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(target_size * 1.14)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.ImageNet(root='./data', split='train', download=False, transform=train_transform)
    val_dataset = datasets.ImageNet(root='./data', split='val', download=False, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=8, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader


def get_celeba_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    # CelebA: using center crop and resize.
    mean = (0.5, 0.5, 0.5)
    std  = (0.5, 0.5, 0.5)
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = datasets.CelebA(root='./data', split='train', download=True, transform=train_transform)
    val_dataset = datasets.CelebA(root='./data', split='valid', download=True, transform=test_transform)
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=train_bs, sampler=train_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, sampler=val_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def get_data_loaders(dataset_name, train_bs, val_bs, orig_size, target_size, use_ddp=False, world_size=1, rank=0):
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return get_mnist_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'cifar10':
        return get_cifar10_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'cifar100':
        return get_cifar100_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'imagenet':
        return get_imagenet_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    elif dataset_name == 'celeba':
        return get_celeba_data_loaders(train_bs, val_bs, orig_size, target_size, use_ddp, world_size, rank)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
