import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED
)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])


def get_datasets():
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    return train_dataset, test_dataset

def get_train_val_datasets():
    train_dataset, test_dataset = get_datasets()
    train_size = int(0.8*len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size,val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders():
    train_dataset, val_dataset, test_dataset = get_train_val_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader, test_loader

