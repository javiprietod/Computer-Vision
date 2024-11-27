# Deep learning libraries
import tensorflow as tf
import numpy as np

# Other libraries
import os

# Own modules (Assuming these have been rewritten for TensorFlow)
from src.tenflow.data import load_tf_data
from src.tenflow.models import CNNModel
from src.tenflow.utils import load_tf_model

# Static variables
SEED = 42
DATA_PATH = "data"
MODELS_PATH = "models"
NUMBER_OF_CLASSES = 10


def main(name: str) -> float:
    """
    This is the main function to evaluate.

    Args:
        name: Name of the model.

    Returns:
        Accuracy on the test set.
    """

    # Load data
    _, _, test_data = load_tf_data(DATA_PATH, batch_size=32)

    # Load model
    model = load_tf_model(name + ".keras")

    # Evaluate
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for batch in test_data:
        # Get the inputs and labels
        inputs, labels = batch

        # Compute predictions
        predictions = model(inputs, training=False)

        # Update accuracy metric
        acc_metric.update_state(labels, predictions)

    acc = acc_metric.result().numpy()
    print(f"Test Accuracy: {acc:.2%}")
    return acc


if __name__ == "__main__":
    main("model_lr_0.00055_hs_(16,)_32_1")
