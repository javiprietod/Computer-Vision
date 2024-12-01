"""
Script to create the Discriminator and Generator for the GAN.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Class to create the Discriminator D of the GAN.
    """

    def __init__(self, out_channels: int = 32) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        out_channels : Number of out channels of the first convolution.
        """

        super().__init__()

        self.out_channels = out_channels
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=out_channels, kernel_size=4, stride=2, padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(out_channels * 2 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Input tensor. Dimensions: [batch, channels, height, width].

        Returns
        -------
        Output tensor. Dimensions: [batch, 1].
        """

        # TODO
        temp = self.conv_1(x)
        temp = self.relu(temp)
        temp = self.conv_2(temp)
        temp = self.bn(temp)
        temp = self.relu(temp)
        temp = temp.view(-1, self.out_channels * 2 * 7 * 7)
        out = self.linear(temp)
        return out


class Generator(nn.Module):
    """
    Class to create the Generator G of the GAN.
    """

    def __init__(self, latent_dim: int, out_channels: int = 32) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        latent_dim   : Dimension of the latent space.
        out_channels : Number of out channels of the first convolution.
        """

        super().__init__()

        self.out_channels = out_channels
        self.linear = nn.Linear(latent_dim, out_channels * 2 * 7 * 7)
        self.bn_1 = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU()
        self.t_conv1 = nn.ConvTranspose2d(
            out_channels * 2, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.t_conv2 = nn.ConvTranspose2d(
            out_channels, 1, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Input tensor. Dimensions: [batch, latent_dim].

        Returns
        -------
        Output tensor. Dimensions: [batch, channels, height, width].
        """

        # TODO
        temp = self.linear(x)
        temp = temp.view(-1, self.out_channels * 2, 7, 7)
        temp = self.bn_1(temp)
        temp = self.relu(temp)
        temp = self.t_conv1(temp)
        temp = self.bn_2(temp)
        temp = self.relu(temp)
        out = self.t_conv2(temp)
        out = torch.sigmoid(out)

        return out


def gan_loss(d_output: torch.Tensor, real_images: bool) -> torch.Tensor:
    """
    Calculates the loss for real (target -> 1) or fake (target -> 0) images.

    Parameters
    ----------
    d_output : Output of the discriminator. Dimensions: [batch, 1].
    real     : True for real images and False for fake images.

    Returns
    -------
    BCE loss with logits.
    """

    # TODO
    target = torch.ones_like(d_output) if real_images else torch.zeros_like(d_output)
    loss = F.binary_cross_entropy_with_logits(d_output, target)
    return loss
