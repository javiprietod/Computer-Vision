# deep learning libraries
import jax
from jax.lib import xla_bridge
from jaxlib.xla_extension import Device

# own modules
from src.data import load_jax_data
from src.jax.training_functions import test_step
from src.utils import load_jax_model

# set device
jax.config.update("jax_default_device", jax.devices("cpu")[0])
device: Device = (
    jax.devices("gpu")[0]
    if xla_bridge.get_backend().platform == "gpu"
    else jax.devices("cpu")[0]
)

# static variables
SEED: int = 42
DATA_PATH: str = "data"
MODELS_PATH: str = "models"
NUMBER_OF_CLASSES: int = 10


def main(name: str) -> float:
    """
    This is the main function to evaluate.

    Args:
        name: name of the model.

    Returns:
        accuracy on the test set.
    """

    # TODO
    # load data
    _, _, test_data = load_jax_data(DATA_PATH, batch_size=32)

    # load model
    model, params = load_jax_model(MODELS_PATH + "/" + name)

    # evaluate
    test_acc = 0.
    for batch in test_data:
        # get the inputs and labels
        inputs, labels = batch

        # compute the accuracy
        acc = test_step(model, params, inputs, labels)
        test_acc += float(acc)
    

    return test_acc / len(test_data)


if __name__ == "__main__":
    print("Accuracy:", main("best_model"))
