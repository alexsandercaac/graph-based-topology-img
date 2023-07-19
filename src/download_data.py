"""
    Script to download datasets using torchvision.
"""
import torchvision

# Data is saved in the data/raw folder
mnist_dataset = torchvision.datasets.MNIST(
    root='data/raw',
    download=True
)
