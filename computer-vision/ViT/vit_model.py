# 3pp
import torch
import torchvision


class VitTransformer(torch.nn.Module):
    """
    This class is to implement a VIT.
    """

    # define attributes
    transformer: torch.nn.Module
    classifier: torch.nn.Module

    def __init__(self, output_size: int = 10, pretrained: bool = True):
        """
        Constructor of Resnet18 class.

        Args:
            output_channels: output size for the model.
            pretrained: bool that indicates if vit is pretrained.
                Defaults to True.
        """

        # TODO
        super(VitTransformer, self).__init__()
        self.transformer = torchvision.models.vit_b_16(pretrained=pretrained)
        self.transformer.requires_grad_(False)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.transformer.heads.head.out_features, output_size),
            # torch.nn.Linear(self.transformer.heads.head.out_features, 500),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.7),
            # torch.nn.Linear(500, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass

        Args:
            inputs: batch of images. Dimensions: [batch, channels,
                height, width].

        Returns:
            batch of logits. Dimensions: [batch, number of classes].
        """

        # TODO
        with torch.no_grad():
            x = self.transformer(inputs)
        x = self.classifier(x)
        return x
