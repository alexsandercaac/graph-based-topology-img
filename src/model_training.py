"""
    Script to train fine tuned models on the datasets
"""
import torchvision
import torch

from utils.dvc.params import get_params


params = get_params()

# Batch size used in training
BATCH_SIZE = params['batch_size']
# Path to the raw data
ROOT = 'data/raw'
# List of datasets to train on
DATASETS = ['mnist', 'cifar10']

for dataset_name in DATASETS:
    print("Dataset: ", dataset_name)
    if dataset_name == 'mnist':
        data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ]),
        }
        dataset_train = torchvision.datasets.MNIST(
            root=ROOT,
            download=True,
            train=True,
            transform=data_transforms['train']
        )
        dataset_val = torchvision.datasets.MNIST(
            root=ROOT,
            download=True,
            train=False,
            transform=data_transforms['val']
        )

    elif dataset_name == 'cifar10':
        data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ]),
        }
        dataset_train = torchvision.datasets.CIFAR10(
            root=ROOT,
            download=True,
            train=True,
            transform=data_transforms['train']
        )
        dataset_val = torchvision.datasets.CIFAR10(
            root=ROOT,
            download=True,
            train=False,
            transform=data_transforms['val']
        )

    image_datasets = {
        'train': dataset_train,
        'val': dataset_val
    }

    # Create dataloaders dictionary for train and validation sets
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        ) for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("Dataset sizes: ", dataset_sizes)
    class_names = image_datasets['train'].classes
    print("Class names: ", class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
