import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    random_split,
)
from torchvision import datasets, transforms

from src.config import (
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
)

# Dataset Statistics
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

# Image Transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# Dataset
def get_datasets() -> tuple[Dataset, Dataset]:
    """
    Load the CIFAR-10 training and test datasets.

    Returns:
        tuple[Dataset, Dataset]:
            Training dataset and test dataset.
    """

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset


# Train / Validation Split
def get_train_val_datasets() -> tuple[
    Subset,
    Subset,
    Dataset,
]:
    """
    Split the training dataset into training and validation sets.

    Returns:
        tuple[Subset, Subset, Dataset]
    """

    train_dataset, test_dataset = get_datasets()

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    return train_dataset, val_dataset, test_dataset


# DataLoaders
def get_dataloaders() -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create DataLoaders for training, validation, and testing.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]
    """

    train_dataset, val_dataset, test_dataset = (
        get_train_val_datasets()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader