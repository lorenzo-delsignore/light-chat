from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from light_chat.data.datamodule import NamesDataModule
from light_chat.models.components.mlp import MLP


class NgramModuleVanilla(nn.Module):
    """Example of a model class for next character prediction using vanilla PyTorch."""

    def __init__(self, net: nn.Module) -> None:
        """Initialize the model with necessary components."""
        super(NgramModuleVanilla, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model"""
        return self.net(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single model step on a batch of data."""
        x = batch
        logits = self.forward(x)
        loss = self.criterion(logits, x["label"])
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step."""
        loss = self.model_step(batch)
        loss.backward()
        for parameters in self.net.parameters():
            parameters.data += -0.01 * parameters.grad
            parameters.grad = None
        self.train_loss.update(loss.item())
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Perform a single validation step."""
        loss = self.model_step(batch)
        self.val_loss.update(loss.item())

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Perform a single test step."""
        loss = self.model_step(batch)
        self.test_loss.update(loss.item())


def train_model(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int
) -> None:
    """Training loop for the model using vanilla PyTorch."""
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            loss = model.training_step(batch)

            print(
                 f"Epoch [{epoch}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}"
            )
        for batch in val_loader:
            model.validation_step(batch)
        print(f"Epoch {epoch} - Training Loss: {model.train_loss.compute()}")
        print(f"Epoch {epoch} - Validation Loss: {model.val_loss.compute()}")
        model.train_loss.reset()
        model.val_loss.reset()


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> None:
    with torch.no_grad():
        for batch in test_loader:
            model.test_step(batch)
    print(f"Test Loss: {model.test_loss.compute()}")
    model.test_loss.reset()


if __name__ == "__main__":
    dataset = NamesDataModule("light_chat/data/names.txt")
    dataset.setup()
    net = MLP()
    model = NgramModuleVanilla(net=net)
    train_model(model, dataset.train_dataloader(), dataset.val_dataloader(), epochs=100)
    evaluate_model(model, dataset.test_dataloader())
