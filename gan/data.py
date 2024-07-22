import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from config import batch_size, dataset_root

mnist_dataset = datasets.MNIST(
    root=dataset_root,
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    ),
    download=True,
)
mnist_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

cifar10_dataset = datasets.CIFAR10(
    root=dataset_root,
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    ),
    download=True,
)
cifar10_dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)
