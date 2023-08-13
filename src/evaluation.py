"""
    Evaluate fine tuned models on the test set
"""
import json

import torchvision
import torch

from utils.dvc.params import get_params
from src.utils.models.modeling import evaluate_model


params = get_params()

# Batch size used in inference
BATCH_SIZE = 4
# Path to the raw data
ROOT = 'data/raw'
# List of datasets to train on
DATASETS = ['mnist', 'cifar10', 'oxford_iit_pet']
metrics_dict = {}
for dataset_name in DATASETS:
    print("Dataset: ", dataset_name)
    if dataset_name == 'mnist':
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ])
        dataset_test = torchvision.datasets.MNIST(
            root=ROOT,
            download=True,
            train=False,
            transform=data_transform
        )

    elif dataset_name == 'cifar10':
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ])
        dataset_test = torchvision.datasets.CIFAR10(
            root=ROOT,
            download=True,
            train=False,
            transform=data_transform
        )

    elif dataset_name == 'oxford_iit_pet':
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ])
        dataset_test = torchvision.datasets.OxfordPets(
            root=ROOT,
            download=True,
            split='test',
            transform=data_transform
        )

    dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    dataset_size = len(dataset_test)
    print("Test dataset size: ", dataset_size)

    # Load the model
    model = torch.load(f'models/{dataset_name}_model.pt')

    # Evaluate the model
    avg_loss, avg_acc = evaluate_model(model, dataloader, dataset_size, params)

    print(f"Average loss: {avg_loss:.4f}, Average accuracy: {avg_acc:.4f}")

    metrics = {
        'avg_loss': avg_loss,
        'avg_acc': avg_acc
    }

    metrics_dict[dataset_name] = metrics

# Write the metrics to a file
with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f)
