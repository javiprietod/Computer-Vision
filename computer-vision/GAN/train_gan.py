"""
Script to train the GAN.
"""

import torch
from torch import optim, nn
from torchvision import transforms, datasets  # type: ignore

from src.gan import Discriminator, Generator, gan_loss
from src.utils import save_generated_samples, save_gan_losses

device = "cuda" if torch.cuda.is_available() else "cpu"


def train() -> tuple[nn.Module, nn.Module]:
    """
    Trains the GAN.

    Returns
    -------
    d : Discriminator.
    g : Generator.
    """

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Training parameters and lists
    d_real_losses = []
    d_fake_losses = []
    d_losses = []
    g_losses = []

    # TODO: Define Discriminator and Generator and complete the training.
    # At the end of each epoch use this part to save the evolution of the generator:
    epochs = 1
    latent_dim = 100
    d = Discriminator()
    g = Generator(latent_dim)
    d.to(device)
    g.to(device)
    d_optimizer = optim.Adam(d.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(g.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real_images, _ in trainloader:
            real_images = real_images.to(device)
            d_optimizer.zero_grad()
            d_output = d(real_images)
            d_real_loss = gan_loss(d_output, real_images=True)
            d_real_loss.backward()
            d_real_losses.append(d_real_loss.item())
            d_optimizer.step()

            d_optimizer.zero_grad()
            z = torch.rand((real_images.shape[0], latent_dim)).to(device)
            fake_images = g(z)
            d_output = d(fake_images)
            d_fake_loss = gan_loss(d_output, real_images=False)
            d_fake_loss.backward()
            d_fake_losses.append(d_fake_loss.item())
            d_optimizer.step()

            d_losses.append(d_real_loss.item() + d_fake_loss.item())

            g_optimizer.zero_grad()
            z = torch.rand((real_images.shape[0], latent_dim)).to(device)
            fake_images = g(z)
            d_output = d(fake_images)
            g_loss = gan_loss(d_output, real_images=True)
            g_loss.backward()
            g_optimizer.step()

            g_losses.append(g_loss.item())
    # g.eval()
    # samples_z = g(fixed_z)
    # samples.append(samples_z)
    # g.train()

    # You can change this part to adjust it to your figures
    max_epochs = 4
    samples_per_epoch = 8
    step = 1 if epochs < 5 else min(max_epochs, epochs // 5)
    epochs_to_check = list(range(0, epochs, step))
    if step > max_epochs and (epochs - 1) % 5 != 0:
        epochs_to_check += [epochs - 1]
    # save_generated_samples(epochs_to_check, samples, samples_per_epoch)
    # save_gan_losses(d_real_losses, d_fake_losses, d_losses, g_losses)

    return d, g
