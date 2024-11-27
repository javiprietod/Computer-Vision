import torch


class Block(torch.nn.Module):
    """
    Neural net block composed of 3x(conv(kernel 3) + ReLU)

    Attr:
        net: neural net
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        """
        Constructor of Block class.

        Args:
            input_channels: input channels for Block.
            output_channels: output channels for Block.
            stride: stride for the second convolution of the Block.

        Returns:
            None.
        """

        # call torch.nn.Module constructor
        super().__init__()

        # fill network
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Method that returns the output of the neural net

        Args:
            inputs: batch of tensors. Dimensions: [batch,
                input_channels, height, width].

        Returns:
            batch of tensors. Dimensions: [batch, output_channels,
                (height - 1)/stride + 1, (width - 1)/stride + 1].
        """

        return self.net(inputs)


class ToyModel(torch.nn.Module):
    """
    Model composed of a cnn_net and a linear classifier at the end

    Attr:
        cnn_net: neural net composed of conv layers, ReLUs and a max
            pooling.
        classifier: a linear layer.
    """

    def __init__(
        self,
        layers: tuple[int, int, int] = (32, 64, 128),
        input_channels: int = 3,
        output_channels: int = 10,
    ):
        """
        Constructor of the class ToyModel.

        Args:
            layers: output channel dimensions of the Blocks.
            input_channels : input channels of the model.

        Returns:
            None.
        """

        # call torch.nn.Module constructor
        super().__init__()

        # initialize module_list
        module_list = [
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        ]

        # add 3 SuperTuxBlocks to module_list
        last_layer = 32
        for layer in layers:
            module_list.append(Block(last_layer, layer, stride=2))
            last_layer = layer
        self.cnn_net = torch.nn.Sequential(*module_list)

        # define GAP
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        # add a final linear layer for classification
        self.classifier = torch.nn.Linear(last_layer, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits. It is the output of the neural network

        Args:
            batch of images. Dimensions: [batch, channels,
                height, width].

        Returns:
            batch of logits. Dimensions: [batch, 6].
        """

        # compute the features
        outputs = self.cnn_net(inputs)
        # GAP
        outputs = self.gap(outputs)

        # flatten output and compute linear layer output
        outputs = torch.flatten(outputs, 1)
        outputs = self.classifier(outputs)

        return outputs
