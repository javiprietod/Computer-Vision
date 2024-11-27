# deep learning libraries
import jax
from flax import linen as nn

# other libraries
from typing import Callable


class Block(nn.Module):
    """
    Neural net block composed of 3x(conv(kernel=3, padding=1) + ReLU).

    Attributes:
        net: sequential containing all the layers.
    """

    input_channels: int
    output_channels: int
    kernel_size: tuple[int, ...] = (3, 3)
    padding: int = 1

    # TODO
    def setup(self) -> None:
        """
        Constructor of Block.

        Args:
            hidden_size: size of the hidden layer.
            kernel_size: size of the kernel.
        """
        # set attributes

        self.net = nn.Sequential(
            [
                nn.Conv(
                    features=self.input_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                ),
                nn.relu,
                nn.Conv(
                    features=self.output_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                ),
                nn.relu,
                nn.Conv(
                    features=self.output_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                ),
                nn.relu,
            ]
        )
        self.max_pool = nn.max_pool

    def __call__(self, x) -> jax.Array:
        """
        This method computes the forward pass.

        Args:
            x: input tensor.

        Returns:
            output tensor.
        """

        # TODO
        x = self.net(x)
        return x


class CNNModel(nn.Module):
    """
    Model constructed used Block modules.
    """

    hidden_sizes: tuple[int, ...]
    input_channels: int = 3
    output_channels: int = 10

    # TODO
    def setup(self) -> None:
        """
        Constructor of CNNModel.

        Args:
            hidden_sizes: sizes of the hidden layers.
        """
        # set attributes

        # create the blocks
        self.conv1 = nn.Conv(
            features=self.hidden_sizes[0],
            kernel_size=(7, 7),
            padding="SAME",
            strides=(2, 2),
        )
        self.relu = nn.relu
        self.blocks = [
            Block(
                output_channels=self.hidden_sizes[i + 1], input_channels=(hidden_size)
            )
            for i, hidden_size in enumerate(self.hidden_sizes[:-1])
        ]
        self.adaptive_pool = nn.avg_pool
        self.max_pool = nn.max_pool
        # create the output layer
        self.output_layer = nn.Dense(features=self.output_channels)

    def __call__(self, x) -> jax.Array:
        """
        This method computes the forward pass.

        Args:
            x: input tensor.

        Returns:
            output tensor.
        """

        # TODO
        x = self.conv1(x)
        x = self.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        for block in self.blocks:
            x = block(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.mean(axis=(1, 2))

        return self.output_layer(x)
