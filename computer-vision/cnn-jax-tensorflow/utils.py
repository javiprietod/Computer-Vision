# deep learning libraries
from flax import linen as nn

# other libraries
import os
import pickle
from typing import Any


def save_jax_model(model: nn.Module, params: dict[str, Any], path: str) -> None:
    """
    This function is to save the jax model.

    Args:
        model: jax model.
        parameters: parameters of the model.
        path: path to save it.

    Returns:
        None.
    """

    # TODO
    if not os.path.exists("models"):
        os.makedirs("models")
    with open(path + ".pkl", "wb") as f:
        pickle.dump((model, params), f)


def load_jax_model(path: str) -> tuple[nn.Module, dict[str, Any]]:
    """
    This function is to load a jax model.

    Args:
        path: path to load from.

    Returns:
        jax model.
        parameters of the model.
    """

    # TODO
    with open(path + ".pkl", "rb") as f:
        model, params = pickle.load(f)
    return model, params
