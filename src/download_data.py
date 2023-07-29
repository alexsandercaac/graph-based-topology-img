"""
    Script to download datasets using torchvision.
"""
import torchvision


ROOT = 'data/raw'
# Data is saved in the data/raw folder
mnist_dataset = torchvision.datasets.MNIST(
    root=ROOT,
    download=True
)

cifar10_dataset = torchvision.datasets.CIFAR10(
    root=ROOT,
    download=True
)

caltech101_dataset = torchvision.datasets.Caltech101(
    root=ROOT,
    download=True
)

oxfordiit_dataset = torchvision.datasets.OxfordIIITPet(
    root=ROOT,
    download=True
)
