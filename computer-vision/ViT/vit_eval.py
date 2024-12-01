import torch
from torch import optim, nn
from torchvision import transforms, datasets  # type: ignore

from src.vit_model import VitTransformer
from src.utils import save_generated_samples, save_gan_losses, load_data, accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> float:
    """
    Evaluates the ViT model

    Returns None

    """

    _, _, test_loader = load_data("data/imagewoof-320")

    model = torch.load("model.pt")

    model.eval()
    accs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            accs.append(accuracy(outputs, labels))
    test_acc = sum(accs) / len(accs)

    return test_acc
