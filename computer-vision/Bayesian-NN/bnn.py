"""
Script to build a Bayesian Neural Network (BNN).
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore


class BayesianLinear(nn.Module):
    """
    Class to build a Bayesian Linear Layer.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        in_dim  : Dimension of the input.
        out_dim : Dimension of the output.
        """

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        # Mean and std of weights
        self.w_mu = nn.Parameter(torch.normal(0, 0.1, size=(out_dim, in_dim)))
        self.w_sigma = nn.Parameter(
            torch.log(torch.exp(torch.tensor(0.1)) - 1)
            + torch.normal(0, 0.1, size=(out_dim, in_dim))
        )
        # Mean and std of biases
        self.b_mu = nn.Parameter(torch.normal(0, 0.1, size=(out_dim,)))
        self.b_sigma = nn.Parameter(
            torch.log(torch.exp(torch.tensor(0.1)) - 1)
            + torch.normal(0, 0.1, size=(out_dim,))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Input tensor. Dimensions: [batch, self.in_dim].

        Returns
        -------
        Output tensor. Dimensions: [batch, self.out_dim].
        """

        # TODO
        # weights = self.w_mu + self.w_sigma * torch.randn_like(self.w_sigma)
        # biases = self.b_mu + self.b_sigma * torch.randn_like(self.b_sigma)
        # out = x @ weights.t() + biases
        # return out
        def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            """
            Reparameterization trick to sample from N(mu, sigma^2) from N(0,1).

            Parameters
            ----------
            mu    : Mean of the distribution.
            sigma : Standard deviation of the distribution.

            Returns
            -------
            Sampled tensor.
            """
            eps = torch.randn_like(sigma)
            return mu + sigma * eps

        weights = reparameterize(self.w_mu, torch.log(1 + torch.exp(self.w_sigma)))
        biases = reparameterize(self.b_mu, torch.log(1 + torch.exp(self.b_sigma)))
        out = x @ weights.t() + biases
        return out


def bayesian_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    bayesian_layers: list[BayesianLinear],
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the customized bayesian loss using KL with the prior of the parameters
    and the cross entropy of the predictions.

    Parameters
    ----------
    y_true          : True labels.
    y_pred          : Predicted probabilities.
    bayesian_layers : List with the bayesian layers of the network.
    beta            : Parameter to be more or less flexible with the prior.

    Returns
    -------
    kl_loss         : KL loss of the prior of the parameters.
    prediction_loss : Cross entropy loss.
    """

    # TODO
    loss = torch.nn.CrossEntropyLoss()
    prediction_loss = loss(y_pred, y_true)

    # compare the weights and biases of the layers with the prior (N(0, 1))
    kl_loss = 0
    for layer in bayesian_layers:
        w_mu = layer.w_mu
        w_sigma = torch.log(1 + torch.exp(layer.w_sigma))
        b_mu = layer.b_mu
        b_sigma = torch.log(1 + torch.exp(layer.b_sigma))
        kl_loss += torch.mean(
            w_mu**2 / w_sigma**2
            + torch.log(w_sigma)
            - torch.log(w_sigma**2)
            - 1 / 2
        )
        kl_loss += torch.mean(
            b_mu**2 / b_sigma**2
            + torch.log(b_sigma)
            - torch.log(b_sigma**2)
            - 1 / 2
        )

    return kl_loss * beta / (2 * len(bayesian_layers)), prediction_loss


class BayesianNN(nn.Module):
    """
    Class to build a Bayesian Neural Network.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: list[int] | None = None,
        p: float = 0.2,
    ) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        in_dim       : Dimension of the input.
        out_dim      : Dimension of the output.
        hidden_sizes : Dimension of the hidden sizes.
        p            : Probability of dropout.
        """

        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        fc_layers: list[nn.Module] = []
        # First layer
        fc_layers.append(BayesianLinear(in_dim, hidden_sizes[0]))
        fc_layers.append(nn.ReLU())
        # Middle layers
        for i in range(len(hidden_sizes) - 1):
            fc_layers.append(BayesianLinear(hidden_sizes[i], hidden_sizes[i + 1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=p))
        # Last layer
        fc_layers.append(BayesianLinear(hidden_sizes[-1], out_dim))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Input tensor. Dimensions: [batch, in_dim].

        Returns:
        Output tensor. Dimensions: [batch, out_dim].
        """

        return self.fc(x)

    def predict(
        self, x: torch.Tensor, n_samples: int = 100, save_fig: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Obtains the mean and std of the predictions of the network.

        Parameters
        ----------
        x         : Input tensor. Dimensions: [1, in_dim]. Note that it is for a
                    singular element and not a batch.
        n_samples : Number of predictions to generate.
        save_fig  : True to save a figure of the distributions and False otherwise.

        Returns
        -------
        Mean and std of the predictions.
        """

        # TODO
        # To save the figure you can use:
        # fig.savefig("images/bnn/histogram_predictions.png")
        predictions = torch.zeros(n_samples, self.fc[-1].out_dim)
        for i in range(n_samples):
            predictions[i] = self.fc(x)
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        return mean_prediction, std_prediction

    def explore_weights(self) -> None:
        """
        Plots a histogram of mu and sigma of the weights and biases of the model.
        """

        # TODO
        # To save the figure you can use:
        # fig.savefig("images/bnn/model_mu_sigma.png")
        weights = []
        biases = []
        for layer in self.fc:
            if isinstance(layer, BayesianLinear):
                weights.append(layer.w_mu.detach().numpy().flatten())
                weights.append(layer.w_sigma.detach().numpy().flatten())
                biases.append(layer.b_mu.detach().numpy().flatten())
                biases.append(layer.b_sigma.detach().numpy().flatten())
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        sns.histplot(weights[0], ax=axs[0, 0])
        sns.histplot(weights[1], ax=axs[0, 1])
        sns.histplot(biases[0], ax=axs[1, 0])
        sns.histplot(biases[1], ax=axs[1, 1])
        plt.show()
        return None
