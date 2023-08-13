"""
    Module with functions to train pytorch models.
"""
import os

import torch

from tempfile import TemporaryDirectory


def train_model(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.device,
        dataloaders: dict,
        dataset_sizes: dict,
        num_epochs: int = 25,
        verbose: bool = True) -> torch.nn.Module:
    """
    Train a pytorch model.

    Args:
        model (torch.nn.Module): Pytorch model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        device (torch.device): Device to use for training.
        dataloaders (dict): Dictionary with train and validation dataloaders.
        dataset_sizes (dict): Dictionary with train and validation dataset
            sizes.
        num_epochs (int, optional): Number of epochs to train. Defaults to 25.
        verbose (bool, optional): Whether to print training progress.
            Defaults to True.

    Returns:
        torch.nn.Module: Trained model.

    """

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        # Initialize accuracy to 0
        best_acc = 0.0

        for epoch in range(num_epochs):
            if verbose:
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                # Running loss and corrects over the epoch
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # Forward pass
                        outputs = model(inputs)
                        # Saturation of the output
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                if verbose:
                    print(
                        f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
        if verbose:
            print(f'Best val Acc: {best_acc:4f}')

        # Load best model weights and return it
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def evaluate_model(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader) -> tuple:
    """
    Evaluate a pytorch model.

    Args:
        model (torch.nn.Module): Pytorch model to evaluate.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to use for evaluation.
        dataloader (torch.utils.data.DataLoader): Dataloader for evaluation.

    Returns:
        tuple: Tuple with loss and accuracy.

    """
    # Initialize loss and corrects
    running_loss = 0.0
    running_corrects = 0

    # Set model to evaluate mode
    model.eval()

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        # Saturation of the output
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Calculate loss and accuracy
    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)

    return loss, acc
