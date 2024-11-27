# deep learning libraries
import jax
from flax import linen as nn
import optax
from jax.lib import xla_bridge
from jaxlib.xla_extension import Device
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter

# other libraries
import pickle
from tqdm.auto import tqdm

# own modules
from src.data import load_jax_data, JaxDataLoader
from src.jax.models import CNNModel
from src.jax.training_functions import train_step, val_step
from src.utils import save_jax_model

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


def main() -> None:
    """
    This function is the main program for the training.
    """

    # TODO
    epochs: int = 15
    lr: float = 1e-3
    batch_size: int = 32
    hidden_sizes: tuple[int, ...] = (50, 40, 30)  # 256 128

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: JaxDataLoader
    val_data: JaxDataLoader
    train_data, val_data, _ = load_jax_data(DATA_PATH, batch_size=batch_size)

    # define name and writer
    name: str = f"model_lr_{lr}_hs_{hidden_sizes}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    inputs: jax.Array = next(iter(train_data))[0]
    model: nn.Module = CNNModel(
        hidden_sizes=hidden_sizes,
        input_channels=inputs.shape[1],
        output_channels=NUMBER_OF_CLASSES,
    )

    # # define loss and optimizer
    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-3)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(SEED), inputs),
        tx=optimizer,
    )

    # train loop
    train_scores = []
    val_scores = []
    for epoch in tqdm(range(epochs)):
        ############
        # Training #
        ############
        train_acc = 0.0
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = batch
            state, loss, acc = train_step(state, imgs, labels)
            train_acc += acc
        train_acc /= len(train_data)
        train_scores.append(train_acc)
        writer.add_scalar("Loss/train", float(loss), epoch)
        writer.add_scalar("Accuracy/train", float(train_acc), epoch)

        ##############
        # Validation #
        ##############
        val_acc = 0.0
        for batch in tqdm(val_data, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = batch
            loss, acc = val_step(state, imgs, labels)
            val_acc += acc
        val_acc /= len(val_data)
        val_scores.append(val_acc)
        print(
            f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc:05.2%}, Validation accuracy: {val_acc:4.2%}"
        )
        writer.add_scalar("Loss/val", float(loss), epoch)
        writer.add_scalar("Accuracy/val", float(val_acc), epoch)

    # save model
    save_jax_model(model, state.params, "models/" + name)

    return None


if __name__ == "__main__":
    main()
