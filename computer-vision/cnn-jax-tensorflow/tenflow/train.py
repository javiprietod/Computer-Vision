# Deep learning libraries
import tensorflow as tf
from keras import layers, optimizers
import numpy as np
from tqdm.auto import tqdm
import os

# Set seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)

# Other variables
DATA_PATH = "data"
MODELS_PATH = "models"
NUMBER_OF_CLASSES = 10

# Assuming that the following modules have been rewritten accordingly
from src.tenflow.data import load_tf_data  # Replace with your TensorFlow data loading function
from src.tenflow.models import CNNModel  # Replace with your TensorFlow CNN model
from src.tenflow.utils import (
    save_tf_model,
)  # Replace with your TensorFlow model saving function
from src.tenflow.training_functions import (
    train_step,
    val_step,
)  # Replace with your TensorFlow training functions


# Main function
def main():
    # Hyperparameters
    epochs = 1
    lr = 5.5e-4
    batch_size = 32
    hidden_sizes = (16,)
    DATA_PATH = "data"

    # Empty nohup.out file
    open("nohup.out", "w").close()

    # Load data
    train_dataset, val_dataset, _ = load_tf_data(DATA_PATH, batch_size=batch_size)

    # Define name and writer
    name = f"tensorflow_model_lr_{lr}_hs_{hidden_sizes}_{batch_size}_{epochs}"
    log_dir = f"logs/{name}"
    writer = tf.summary.create_file_writer(log_dir)

    # Define model
    model = CNNModel(hidden_sizes=hidden_sizes, output_channels=NUMBER_OF_CLASSES)

    # Define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Define metrics
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    # Training loop
    for epoch in range(epochs):
        # Reset metrics at the start of each epoch
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        # Training step
        for images, labels in tqdm(
            train_dataset, desc=f"Epoch {epoch+1}/{epochs} - Training"
        ):
            train_step(
                model,
                images,
                labels,
                loss_object,
                optimizer,
                train_loss,
                train_accuracy,
            )

        # Validation step
        for val_images, val_labels in tqdm(
            val_dataset, desc=f"Epoch {epoch+1}/{epochs} - Validation"
        ):
            val_step(model, val_images, val_labels, loss_object, val_loss, val_accuracy)

        # Log to TensorBoard
        with writer.as_default():
            tf.summary.scalar("Loss/Train", train_loss.result(), step=epoch)
            tf.summary.scalar("Accuracy/Train", train_accuracy.result(), step=epoch)
            tf.summary.scalar("Loss/Val", val_loss.result(), step=epoch)
            tf.summary.scalar("Accuracy/Val", val_accuracy.result(), step=epoch)

        template = "Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                val_loss.result(),
                val_accuracy.result() * 100,
            )
        )

    # Save model
    save_tf_model(model, name)


if __name__ == "__main__":
    main()
