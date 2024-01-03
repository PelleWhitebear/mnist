# data.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def mnist():
    """Return train and test dataloaders for MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(mnist_dataset))
    test_size = len(mnist_dataset) - train_size
    train_set, test_set = random_split(mnist_dataset, [train_size, test_size])

    # Create DataLoader instances for training and testing sets
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, test_loader
