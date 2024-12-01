import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import math


def spearman_rank_correlation(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Calculate the Spearman Rank Correlation between two tensors.

    Args:
        tensor1: First tensor. Shape: [batch, channels, height, width].
        tensor2: Second tensor. Shape: [batch, channels, height, width].

    Returns:
        corr: Spearman Rank Correlation. Shape: [batch, channels].
    """
    # Flatten the tensors
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")

    while tensor1.dim() < 4:
        tensor1 = tensor1.unsqueeze(0)
        tensor2 = tensor2.unsqueeze(0)

    tensor1 = tensor1.flatten(start_dim=2)
    tensor2 = tensor2.flatten(start_dim=2)

    # Calculate the mean of the tensors
    mean1 = tensor1.mean(dim=2, keepdim=True)
    mean2 = tensor2.mean(dim=2, keepdim=True)

    # Calculate the standard deviation of the tensors
    std1 = tensor1.std(dim=2)

    std2 = tensor2.std(dim=2)

    # Calculate the covariance of the tensors
    cov = ((tensor1 - mean1) * (tensor2 - mean2)).mean(dim=2)

    # Calculate the Spearman Rank Correlation
    corr = cov / (std1 * std2)

    return corr  # Shape: [batch, channels]


def absolute_spearman_correlation(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    abs_corr = torch.abs(spearman_rank_correlation(tensor1, tensor2))
    return abs_corr  # Shape: [batch, channels]


def structural_similarity_index(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Calculate the Structural Similarity Index between two tensors.

    Args:
        tensor1: First tensor. Shape: [batch, channels, height, width].
        tensor2: Second tensor. Shape: [batch, channels, height, width].

    Returns:
        ssim_val: Structural Similarity Index. Shape: [batch, channels].
    """
    # Constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Flatten the tensors
    tensor1 = tensor1.flatten(start_dim=2)
    tensor2 = tensor2.flatten(start_dim=2)

    # Calculate the mean of the tensors
    mu1 = tensor1.mean(dim=2, keepdim=True)
    mu2 = tensor2.mean(dim=2, keepdim=True)

    # Calculate the variance of the tensors
    sigma1 = tensor1.var(dim=2, keepdim=True)
    sigma2 = tensor2.var(dim=2, keepdim=True)

    # Calculate the covariance of the tensors
    sigma12 = ((tensor1 - mu1) * (tensor2 - mu2)).mean(dim=2, keepdim=True)

    # Calculate the SSIM
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
    ssim_val = num / den

    return ssim_val.squeeze(-1)  # Shape: [batch, channels]


def histogram_of_oriented_gradients_sim(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Calculate the Histogram of Oriented Gradients similarity between two tensors.

    Args:
        tensor1: First tensor. Shape: [batch, channels, height, width].
        tensor2: Second tensor. Shape: [batch, channels, height, width].

    Returns:
        similarity: Histogram of Oriented Gradients similarity. Shape: [batch, channels].
    """
    # Constants
    eps = 1e-8

    # # Flatten the tensors
    # tensor1 = tensor1.flatten(start_dim=2)
    # tensor2 = tensor2.flatten(start_dim=2)

    depthwise_kernels = torch.tensor(
        [
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
        ],
        dtype=torch.float32,
    )  # Shape: [3, 3]

    # Reshape to [out_channels, in_channels/groups, kernel_size, kernel_size]
    # For depthwise, out_channels = in_channels and in_channels/groups = 1
    weight = depthwise_kernels.view(3, 1, 3, 3)

    # Perform convolution with groups=in_channels for depthwise operation
    grad1 = F.conv2d(input=tensor1, weight=weight, stride=1, padding=1, groups=3)
    grad2 = F.conv2d(input=tensor2, weight=weight, stride=1, padding=1, groups=3)

    # Calculate the magnitude of the gradients
    mag1 = torch.sqrt(grad1**2 + grad1**2 + eps)
    mag2 = torch.sqrt(grad2**2 + grad2**2 + eps)

    # Calculate the orientation of the gradients
    angle1 = torch.atan2(grad1, grad1)
    angle2 = torch.atan2(grad2, grad2)

    # Calculate the similarity
    similarity = (mag1 * mag2 * torch.cos(angle1 - angle2)) / (mag1 * mag2 + eps)

    return similarity.mean(dim=(2, 3))  # Shape: [batch, channels]
