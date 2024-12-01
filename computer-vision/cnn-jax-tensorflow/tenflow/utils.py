# Deep learning libraries
import keras
from keras import models
import numpy as np

# Other libraries
import os
from typing import Any

from src.tenflow.models import CNNModel, Block


def save_tf_model(model: keras.Model, path: str) -> None:
    """
    This function saves the TensorFlow Keras model.

    Args:
        model: TensorFlow Keras model.
        path: Path to save the model.

    Returns:
        None.
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    models.save_model(model, os.path.join("models", path) + ".keras")


def load_tf_model(path: str) -> keras.Model:
    """
    This function loads a TensorFlow Keras model.

    Args:
        path: Path to load the model from.

    Returns:
        The loaded TensorFlow Keras model.
    """
    model = models.load_model(
        os.path.join("models", path),
        custom_objects={"CNNModel": CNNModel, "Block": Block},
    )
    return model
