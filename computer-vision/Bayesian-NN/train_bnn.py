"""
Script to train a Bayesian Neural Network (BNN).
"""

import torch
from torch import optim, nn
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

from src.bnn import BayesianNN, BayesianLinear, bayesian_loss


def train() -> nn.Module:
    """
    Main function to train the model.

    Returns
    -------
    model : Trained BNN.
    """

    # Load model and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = torch.tensor(pd.read_csv("data/X_train.csv").values, dtype=torch.float32)
    x_test = torch.tensor(pd.read_csv("data/X_test.csv").values, dtype=torch.float32)
    y_train = torch.tensor(pd.read_csv("data/y_train.csv").values, dtype=torch.float32)
    y_train = torch.tensor([int(x.item()) for x in y_train])
    y_test = torch.tensor(pd.read_csv("data/y_test.csv").values, dtype=torch.float32)
    y_test = torch.tensor([int(x.item()) for x in y_test])
    n_features = x_train.shape[1]
    n_classes = 3
    # To store training data
    # Note that there is only one KL list because they are the same in train and test
    # (in test parameters are not updated).
    kl_losses = []
    prediction_losses_train = []
    train_losses = []
    train_accs = []
    prediction_losses_test = []
    test_losses = []
    test_accs = []

    # TODO: Define the model and complete the training.
    # 1. Define the model
    model = BayesianNN(n_features, n_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    epochs = 239
    beta = 0.18

    all_layers = model.fc
    bayesian_layers = [
        layer for layer in all_layers if isinstance(layer, BayesianLinear)
    ]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        kl_loss, prediction_loss = bayesian_loss(
            y_train, y_pred, bayesian_layers, beta=beta
        )
        loss = kl_loss + prediction_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()

            kl_loss, prediction_loss = bayesian_loss(
                y_train, y_pred, bayesian_layers, beta=beta
            )
            train_acc = (y_pred.argmax(dim=1) == y_train).float().mean()
            train_losses.append(loss.item())
            train_accs.append(train_acc.item())

            y_pred = model(x_test)
            kl_loss, prediction_loss = bayesian_loss(
                y_test, y_pred, bayesian_layers, beta=beta
            )
            test_loss = kl_loss + prediction_loss
            test_acc = (y_pred.argmax(dim=1) == y_test).float().mean()
            test_losses.append(test_loss.item())
            test_accs.append(test_acc.item())

            kl_losses.append(kl_loss.item())
            prediction_losses_train.append(prediction_loss.item())
            prediction_losses_test.append(prediction_loss.item())
        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

    # Save training evolution
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    x = range(1, epochs + 1)
    if isinstance(axs, np.ndarray):  # to pass mypy
        axs[0].plot(x, prediction_losses_train, label="Prediction train")
        axs[0].plot(x, kl_losses, label="KL")
        axs[0].plot(x, prediction_losses_test, label="Prediction Test")
        axs[0].plot(x, train_losses, label="Total Train")
        axs[0].plot(x, test_losses, label="Total Test")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[1].plot(x, train_accs, label="Train")
        axs[1].plot(x, test_accs, label="Test")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
    fig.savefig("images/bnn/train.png")
    plt.close(fig)

    return model
