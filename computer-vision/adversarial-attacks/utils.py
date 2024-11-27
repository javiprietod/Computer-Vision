import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import io
import contextlib


CIFAR_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _add_information(
    img: torch.Tensor, perturbed_img: torch.Tensor, label: int, model: nn.Module
) -> str:
    """
    Adds information to the image corresponding to the "before and after" of the
    perturbation.

    Parameters
    ----------
    img           : Original image. Dimensions: [channels, height, width].
    perturbed_img : Perturbed image. Dimensions: [channels, height, width].
    label         : Real label of the image.
    model         : Model used.

    Returns
    -------
    String with the included information.
    """

    real_class = CIFAR_LABELS[label]
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        print(f"True Label: {real_class}")
        with torch.no_grad():
            y_pred = F.softmax(model(img.unsqueeze(0)).squeeze(), dim=0)
            y_pred_perturbed = F.softmax(
                model(perturbed_img.unsqueeze(0)).squeeze(), dim=0
            )

        original_prediction = CIFAR_LABELS[torch.argmax(y_pred)]
        original_prob_label = y_pred[label].item()
        original_prob = torch.max(y_pred).item()
        print(f"\nOriginal prediction: {original_prediction}")
        print(
            f"Probability for original label ({real_class}):"
            f"{original_prob_label:.2f}"
        )
        print(
            f"Probability for original prediction ({original_prediction}):"
            f"{original_prob:.2f}"
        )

        perturbed_prediction = CIFAR_LABELS[torch.argmax(y_pred_perturbed)]
        perturbed_prob_label = y_pred_perturbed[label].item()
        perturbed_prob = torch.max(y_pred_perturbed).item()
        print(f"\nPerturbed prediction: {perturbed_prediction}")
        print(
            f"Probability for original prediction ({original_prediction}):"
            f"{perturbed_prob_label:.2f}"
        )
        print(
            f"Probability for perturbed prediction ({perturbed_prediction}):"
            f"{perturbed_prob:.2f}"
        )

    return output.getvalue()


def visualize_perturbations(
    perturbed_img: torch.Tensor,
    img: torch.Tensor,
    label: int,
    model: nn.Module,
    title: str | None = None,
) -> None:
    """
    Saves a figure with the "before and after" of the perturbation.

    Parameters
    ----------
    perturbed_img : Perturbed image. Dimensions: [channels, height, width].
    img           : Original image. Dimensions: [channels, height, width].
    label         : Real label of the image.
    model         : Model used.
    title         : Title of the figure.
    """

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(
        _add_information(img, perturbed_img, label, model),
        ha="center",
        fontsize=14,
        fontfamily="monospace",
        wrap=True,
        y=1.05,
    )

    axs[0].imshow(
        np.transpose(img.cpu().numpy(), (1, 2, 0)), interpolation="nearest"
    )  # type: ignore
    axs[0].set_title("Original Image")  # type: ignore
    axs[1].imshow(
        np.transpose(perturbed_img.cpu().numpy(), (1, 2, 0)),
        interpolation="nearest",
    )  # type: ignore
    axs[1].set_title("Perturbed Image")  # type: ignore
    plt.subplots_adjust(top=0.8)

    if title is None:
        title = f"adversarial_attack_{CIFAR_LABELS[label]}"
    plt.savefig(f"images/{title}.png", bbox_inches="tight")


def save_img_cutmix(
    images: torch.Tensor,
    labels: tuple[int, int],
    cutmix_img: torch.Tensor,
    cutmix_label: float,
    title: str | None = None,
) -> None:
    """
    Saves a figure for the "before and after" of the CutMix.

    Parameters
    ----------
    images       : Original images. Dimensions: [2, channels, height, width].
    labels       : Original labels.
    cutmix_img   : Modified image. Dimensions: [channels, height, width].
    cutmix_label : Modified label.
    """

    _, axs = plt.subplots(1, 3, figsize=(14, 8))

    axs[0].imshow(
        np.transpose(images[0].cpu().numpy(), (1, 2, 0)), interpolation="nearest"
    )  # type: ignore
    axs[0].set_title(f"Original Image. Label: {labels[0]}.")  # type: ignore

    axs[1].imshow(
        np.transpose(images[1].cpu().numpy(), (1, 2, 0)),
        interpolation="nearest",
    )  # type: ignore
    axs[1].set_title(f"Original Image. Label: {labels[1]}.")  # type: ignore

    axs[2].imshow(
        np.transpose(cutmix_img.cpu().numpy(), (1, 2, 0)),
        interpolation="nearest",
    )  # type: ignore
    axs[2].set_title(f"CutMix Image. Label: {cutmix_label:.2f}.")  # type: ignore

    if title is None:
        title = "CutMix"
    plt.savefig(f"images/{title}.png", bbox_inches="tight")


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Parameters
    ----------
    seed : Seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
