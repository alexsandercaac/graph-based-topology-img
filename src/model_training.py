"""
    Script to train fine tuned models on the datasets
"""
import torchvision
import torch

from torch import nn

from utils.dvc.params import get_params
from src.utils.models.modeling import train_model


params = get_params()

# Batch size used in training
BATCH_SIZE = params['batch_size']
# Number of epochs to train
NUM_EPOCHS = params['num_epochs']
# Initial learning rate
INITIAL_LR = params['initial_lr']
# Learning rate decay factor
LR_DECAY_FACTOR = params['lr_decay_factor']
# Number of epochs after which to decay the learning rate
LR_DECAY_STEP = params['lr_decay_step']
# Path to the raw data
ROOT = 'data/raw'
# List of datasets to train on
DATASETS = ['mnist', 'cifar10', 'oxford_iit_pet']

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

        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train,
            [0.8, 0.2]
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
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train,
            [0.8, 0.2]
        )

    elif dataset_name == 'oxford_iit_pet':
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
        dataset_train = torchvision.datasets.OxfordPets(
            root=ROOT,
            download=True,
            split='trainval',
            transform=data_transforms['train']
        )
        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train,
            [0.8, 0.2]
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

    # Load the pretrained model
    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    # The mnist dataset has only 1 channel, so the first layer of the model
    # needs to be changed to accept 1 channel instead of 3
    if dataset_name == 'mnist':
        model_conv.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_conv = torch.optim.Adam(
        model_conv.fc.parameters(),
        lr=INITIAL_LR
    )

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_conv,
        step_size=LR_DECAY_STEP,
        gamma=LR_DECAY_FACTOR
    )

    model_conv = train_model(
        model_conv,
        criterion,
        optimizer_conv,
        exp_lr_scheduler,
        device,
        dataloaders,
        dataset_sizes,
        NUM_EPOCHS
    )

    torch.save(model_conv.state_dict(), f'models/{dataset_name}_model.pt')
