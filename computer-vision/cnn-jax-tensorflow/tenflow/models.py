# Deep learning libraries
import tensorflow as tf
from keras import layers, models

from keras.utils import register_keras_serializable

# Other libraries
from typing import Callable, List, Tuple


@register_keras_serializable()
class Block(tf.keras.layers.Layer):
    """
    Neural net block composed of 3x(conv(kernel=3, padding=1) + ReLU).
    """

    def __init__(
        self,
        input_shape: int,
        output_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        padding: str = "same",
        **kwargs
    ):
        super(Block, self).__init__()
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv1 = layers.Conv2D(
            filters=input_shape, kernel_size=kernel_size, padding=padding
        )
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(
            filters=output_channels, kernel_size=kernel_size, padding=padding
        )
        self.relu2 = layers.ReLU()
        self.conv3 = layers.Conv2D(
            filters=output_channels, kernel_size=kernel_size, padding=padding
        )
        self.relu3 = layers.ReLU()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x

    def get_config(self):
        config = super(Block, self).get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "output_channels": self.output_channels,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
            }
        )
        return config


@register_keras_serializable()
class CNNModel(tf.keras.Model):
    def __init__(
        self,
        hidden_sizes: Tuple[int, ...],
        input_channels: int = 3,
        output_channels: int = 10,
        **kwargs
    ):
        super(CNNModel, self).__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Define layers here if input shape is known
        self.conv1 = layers.Conv2D(
            filters=self.hidden_sizes[0],
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="same",
        )
        self.relu = layers.ReLU()
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.blocks = []
        for i in range(len(self.hidden_sizes) - 1):
            block = Block(
                input_shape=self.hidden_sizes[i],
                output_channels=self.hidden_sizes[i + 1],
            )
            self.blocks.append(block)
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.output_layer = layers.Dense(units=self.output_channels)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
            x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        config = super(CNNModel, self).get_config()
        config.update(
            {
                "hidden_sizes": self.hidden_sizes,
                "input_channels": self.input_channels,
                "output_channels": self.output_channels,
            }
        )
        return config
