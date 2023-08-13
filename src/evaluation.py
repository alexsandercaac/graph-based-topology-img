"""
    Evaluate fine tuned models on the test set
"""
import json

import torchvision
import torch

from utils.models.modeling import evaluate_model


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
        dataset_test = torchvision.datasets.OxfordIIITPet(
            root=ROOT,
            download=True,
            split='test',
            transform=data_transform
        )
    class_names = dataset_test.classes
    dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    dataset_size = len(dataset_test)
    print("Test dataset size: ", dataset_size)

    # Load the model
    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = torch.nn.Linear(num_ftrs, len(class_names))
    # The mnist dataset has only 1 channel, so the first layer of the model
    # needs to be changed to accept 1 channel instead of 3
    if dataset_name == 'mnist':
        model_conv.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_state = torch.load(f'models/{dataset_name}_model.pt')

    model_conv.load_state_dict(model_state)
    print("Successfully loaded trained model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss()
    # Evaluate the model
    model_conv.to(device)
    avg_loss, avg_acc = evaluate_model(
        model=model_conv,
        criterion=criterion,
        device=device,
        dataloader=dataloader
    )

    print(f"Average loss: {avg_loss:.4f}, Average accuracy: {avg_acc:.4f}")

    metrics = {
        'avg_loss': avg_loss,
        'avg_acc': avg_acc.tolist()
    }

    metrics_dict[dataset_name] = metrics
print(metrics_dict)
# Write the metrics to a file
with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics_dict, f)
