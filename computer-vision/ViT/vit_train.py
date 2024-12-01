import torch
from torch import optim, nn
from torchvision import transforms, datasets  # type: ignore
from tqdm import tqdm

from src.vit_model import VitTransformer
from src.utils import (
    save_generated_samples,
    save_gan_losses,
    ImagewoofDataset,
    load_data,
    download_data,
    accuracy,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


def train() -> None:
    """
    Trains the ViT and saves the model

    Returns None

    """

    # Data
    download_data("data/")
    train_loader, val_loader, test_loader = load_data("data/imagewoof-320")
    # Training parameters and lists
    train_losses = []

    model = VitTransformer()
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(10)):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            print(i / len(train_loader), end="\r")
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        print("\n")
        scheduler.step()

        model.eval()
        accs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                accs.append(accuracy(outputs, labels))
        val_acc = sum(accs) / len(accs)
        print(val_acc)

    torch.save(model, "model.pt")

    return None


if __name__ == "__main__":
    train()
