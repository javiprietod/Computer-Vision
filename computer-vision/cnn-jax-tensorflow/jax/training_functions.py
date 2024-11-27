# deep learning libraries
import jax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

# other libraries
from typing import Any


@jax.jit
def train_step(
    state: TrainState, inputs: jax.Array, labels: jax.Array
) -> tuple[TrainState, jax.Array, jax.Array]:
    """
    This function computes the training step.

    Args:
        state: training state.
        inputs: inputs for the model. Dimensions:
            [batch, height, width, channels].
        labels: labels for the training. Dimensions: [batch, 1].

    Returns:
        trained state.
        loss value.
        accuracy value.
    """

    # TODO
    grad_fn = jax.value_and_grad(metrics, has_aux=True)
    (loss, acc), grads = grad_fn(state.params, state, inputs, labels)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


@jax.jit
def val_step(
    state: TrainState, inputs: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    This function computes the training step.

    Args:
        state: training state.
        inputs: inputs for the model. Dimensions:
            [batch, height, width, channels].
        labels: labels for the training. Dimensions: [batch, 1].

    Returns:
        loss value.
        accuracy value.
    """

    # TODO
    loss, acc = metrics(state.params, state, inputs, labels)
    return loss, acc


# @jax.jit
def test_step(
    model: nn.Module, parameters: dict[str, Any], inputs: jax.Array, labels: jax.Array
) -> jax.Array:
    """
    This function has to compute the accuracy of the predictions.

    Args:
        model: jax model.
        inputs: inputs of the model.
        labels: labels to predict.

    Returns:
        accuracy of the model.
    """

    # TODO
    logits = model.apply(parameters, inputs)
    # acc = (labels == logits.argmax(axis=-1)).mean()
    if isinstance(logits, dict):
        logits = logits['logits']
    acc = jax.numpy.mean(labels == jax.numpy.argmax(logits, -1))
    return acc


def metrics(params, state, inputs: jax.Array, labels) -> tuple[jax.Array, jax.Array]:
    """
    This function computes the loss and the accuracy of the model.

    Args:
        params:
        state: _description_
        images: _description_
        labels: _description_

    Returns:
        _description_
    """

    # TODO

    logits = state.apply_fn(params, inputs)
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    acc = (labels == logits.argmax(axis=-1)).mean()
    return loss, acc
