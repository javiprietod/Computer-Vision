import torch
import numpy as np


def cutmix(
    images: torch.Tensor, labels: tuple[int, int], alpha: float = 1.0
) -> tuple[torch.Tensor, float]:
    """
    Applies CutMix to a pair of images.

    Parameters
    ----------
    inputs : Pair of images. Dimensions: [2, channels, height, width].
    labels : Labels of both images.
    alpha  : Parameter for the beta distribution.

    Returns
    -------
    x_cutmix : Modified image. Dimensions: [channels, height, width].
    y_cutmix : Modified label.
    """

    # TODO
    lamda = np.random.beta(alpha, alpha)

    # create a bounding box
    height = images.size(2)
    width = images.size(3)

    rx, ry, rw, rh = (
        np.random.randint(width),
        np.random.randint(height),
        width * np.sqrt(1 - lamda),
        height * np.sqrt(1 - lamda),
    )

    mask = torch.zeros_like(images[0])
    mask[:, rx : int(rx + rw), ry : int(ry + rh)] = 1

    x_cutmix = images[0] * mask + images[1] * (1 - mask)
    y_cutmix = lamda * labels[0] + (1 - lamda) * labels[1]

    return x_cutmix, y_cutmix
