"""
Script for some utility functions.
"""

# standard libraries
import os
import shutil
import random
import tarfile
from tarfile import TarFile
import urllib.request

# 3pp
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from requests.models import Response


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


def save_generated_samples(
    epochs: list[int], samples: list[torch.Tensor], samples_per_epoch: int = 8
) -> None:
    """
    Creates a figure with the same latent values but in different training epochs.

    Parameters
    ----------
    epochs            : List with the epochs we want to see.
    samples           : List in which each element is a batch of generated images of
                        dimensions [batch, channels, height, width].
    samples_per_epoch : Number of samples to show per epoch.
    """

    fig, axs = plt.subplots(
        len(epochs),
        samples_per_epoch,
        figsize=(4 * len(epochs), samples_per_epoch),
    )
    fig.tight_layout(rect=(0.1, 0, 1, 0.95))
    fig.suptitle("Images Generated", fontsize=35)
    for i, epoch in enumerate(epochs):
        for j, img in enumerate(samples[epoch][:samples_per_epoch]):
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            axs[i, j].xaxis.set_visible(False)
            axs[i, j].yaxis.set_visible(False)
            axs[i, j].imshow(img, cmap=plt.get_cmap("gray"))
        fig.text(
            0.04,
            1 - (i + 0.5) / len(epochs),
            f"Epoch {epoch}",
            ha="center",
            va="center",
            fontsize=25,
        )

    fig.savefig("images/gan/generated_samples_per_epoch.png")
    plt.close(fig)


def save_gan_losses(
    d_real_losses: list[float],
    d_fake_losses: list[float],
    d_losses: list[float],
    g_losses: list[float],
) -> None:
    """
    Saves the evolution of the losses in the GAN.

    Parameters
    ----------
    d_real_losses : Losses of the discriminator when predicting real images.
    d_fake_losses : Losses of the discriminator when predicting fake images.
    d_losses      : Losses of the discriminator when predicting real and fake images
                    (is the sum of d_real_losses and d_fake_losses).
    g_losses      : Losses of the generator.
    """

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(d_real_losses, linewidth=0.5, label="Total")
    axs[0].plot(d_fake_losses, linewidth=0.5, label="Fake")
    axs[0].plot(d_losses, linewidth=0.5, label="Total")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Discriminator")
    axs[0].legend()

    axs[1].plot(g_losses)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Generator")

    fig.savefig("images/gan/gan_losses.png")
    plt.close(fig)


class ImagewoofDataset(Dataset):
    """
    This class is the Imagewoof Dataset.
    """

    # TODO
    def __init__(self, root_dir: str, split: str = "train", transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            split (str): 'train', 'val', or 'test'.
            transform: Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    self.images.append(
                        (os.path.join(cls_dir, img_name), self.class_to_idx[cls])
                    )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data(
    path: str, batch_size: int = 64, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for imagewoof dataset.

    Args:
        path: path of the dataset.
        color_space: color_space for loading the images.
        batch_size: batch size for dataloaders. Default value: 128.and
        num_workers: number of workers for loading data.
            Default value: 0.

    Returns:
        tuple of dataloaders, train, val and test in respective order.
    """

    # TODO
    train_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]
    )

    # Create datasets
    train_dataset = ImagewoofDataset(
        root_dir=path, split="train", transform=train_transform
    )

    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2])

    val_dataset = ImagewoofDataset(
        root_dir=path, split="val", transform=val_test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_loader, val_loader, test_loader


def download_data(path: str) -> None:
    """
    This function downloads the data from internet.

    Args:
        path: path to dave the data.
    """

    # TODO
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-320.tgz"
    filename = os.path.join(path, "imagewoof-320.tgz")
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(filename):
        print(f"Downloading Imagewoof dataset from {url}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print("Imagewoof dataset archive already exists.")

    # Extract the dataset
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=path)
    print("Extraction complete.")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """
    This method computes accuracy from logits and labels

    Args:
        logits: batch of logits. Dimensions: [batch, number of classes]
        labels: batch of labels. Dimensions: [batch]

    Returns:
        accuracy of predictions
    """

    # compute predictions
    predictions = logits.argmax(1).type_as(labels)

    # compute accuracy from predictions
    result = predictions.eq(labels).float().mean().cpu().detach().numpy()

    return result
