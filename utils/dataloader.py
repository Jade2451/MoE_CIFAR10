import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """
    Creates and returns the training and testing DataLoaders for CIFAR-10.

    Args:
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        A tuple containing the training DataLoader and testing DataLoader.
    """
    
    # Define data augmentations for the training set
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

    # Define transformations for the validation/test set (only normalization)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    print(f"Data loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    return train_loader, test_loader