from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric


class TrigramModuleVanilla(nn.Module):
    """Example of a model class for next character prediction using vanilla PyTorch."""

    def __init__(self) -> None:
        """Initialize the model with necessary components."""
        super(TrigramModuleVanilla, self).__init__()
        self.criterion = nn.NLLLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.W = nn.Parameter(
            torch.randn((601, 27), generator=torch.Generator().manual_seed(42))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model"""
        bigram_enc = F.one_hot(x["bigram_idx"], num_classes=601).float()
        logits = bigram_enc @ self.W
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)
        return probs

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single model step on a batch of data."""
        x = batch
        logits = self.forward(x)
        loss = self.criterion(torch.log(logits), x["label_idx"])
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step."""
        loss = self.model_step(batch)
        self.W.grad = None
        loss.backward()
        self.W.data += -0.8 * self.W.grad + 0.0005 * (self.W**2).mean()
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
        # Log metrics for the epoch
        print(f"Epoch {epoch} - Training Loss: {model.train_loss.compute()}")
        print(f"Epoch {epoch} - Validation Loss: {model.val_loss.compute()}")
        torch.save(model.W, "W_parameter.pth")
        # Reset metrics after each epoch
        model.train_loss.reset()
        model.val_loss.reset()


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> None:
    with torch.no_grad():
        for batch in test_loader:
            model.test_step(batch)
    # Log test metrics
    print(f"Test Loss: {model.test_loss.compute()}")
    # Reset metrics after evaluation
    model.test_loss.reset()
