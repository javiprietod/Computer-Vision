# Deep learning libraries
import tensorflow as tf
import numpy as np

# Other libraries
import os
import math
import requests
import tarfile
import shutil
from PIL import Image
from typing import Tuple
from requests.models import Response
from tarfile import TarFile
from torchvision import transforms


def download_data(path: str) -> None:
    """
    This function downloads and processes the Imagenette dataset.

    Args:
        path: Path where the dataset will be downloaded and processed.
    """
    # Define paths
    url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    target_path: str = f"{path}/imagenette2.tgz"

    if not os.path.exists(path):
        os.makedirs(path)

    # download tar file
    response: Response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, "wb") as f:
            f.write(response.raw.read())

    # extract tar file
    tar_file: TarFile = tarfile.open(target_path)
    tar_file.extractall(path)
    tar_file.close()
    # create final save directories
    os.makedirs(f"{path}/train")
    os.makedirs(f"{path}/val")

    # define resize transformation
    transform = transforms.Resize((224, 224))

    # loop for saving processed data
    for split in ("train", "val"):
        list_class_dirs = os.listdir(f"{path}/imagenette2/{split}")
        for j in range(len(list_class_dirs)):
            # if list_class_dirs[j] == ".DS_Store":
            #     continue
            list_dirs = os.listdir(f"{path}/imagenette2/{split}/{list_class_dirs[j]}")
            for k in range(len(list_dirs)):
                # if list_dirs[k] == ".DS_Store":
                #     continue
                image = Image.open(
                    f"{path}/imagenette2/{split}/"
                    f"{list_class_dirs[j]}/{list_dirs[k]}"
                )
                image = transform(image)
                numpy_image = np.array(image)

                if numpy_image.shape == (224, 224, 3):
                    np.save(f"{path}/{split}/{j}_{k}.npy", numpy_image)

    # delete other files
    os.remove(target_path)
    shutil.rmtree(f"{path}/imagenette2")

    return None


def tf_load_image(file_path, label):
    """
    Loads an image and label using TensorFlow operations.

    Args:
        file_path: Path to the .npy image file.
        label: Label associated with the image.

    Returns:
        A tuple of the image tensor and label.
    """

    def load_image(path):
        image_np = np.load(path.decode()).astype(np.float32)
        return image_np

    image = tf.numpy_function(func=load_image, inp=[file_path], Tout=tf.float32)
    image.set_shape([224, 224, 3])
    return image, label


def create_dataset_from_indices(path, indices):
    """
    Creates a TensorFlow dataset from given indices.

    Args:
        path: Path to the dataset.
        indices: List of indices to include in the dataset.

    Returns:
        A TensorFlow Dataset object.
    """
    names = sorted(os.listdir(path))
    selected_names = [names[i] for i in indices]
    file_paths = [os.path.join(path, name) for name in selected_names]
    labels = [int(name.split("_")[0]) for name in selected_names]
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(tf_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def load_tf_data(
    path: str, batch_size: int = 128
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the data and returns TensorFlow datasets for training, validation, and testing.

    Args:
        path: Path where the dataset is located.
        batch_size: Batch size for the datasets.

    Returns:
        A tuple containing the training, validation, and test datasets.
    """
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "val")

    # Get list of training file names
    train_names = sorted(os.listdir(train_path))
    train_file_paths = [os.path.join(train_path, name) for name in train_names]
    train_labels = [int(name.split("_")[0]) for name in train_names]

    # Shuffle indices and split into training and validation sets
    dataset_size = len(train_file_paths)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    train_size = int(0.7 * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create datasets
    train_dataset = create_dataset_from_indices(train_path, train_indices)
    val_dataset = create_dataset_from_indices(train_path, val_indices)

    # Process test dataset
    test_names = sorted(os.listdir(test_path))
    test_file_paths = [os.path.join(test_path, name) for name in test_names]
    test_labels = [int(name.split("_")[0]) for name in test_names]
    test_dataset = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))
    test_dataset = test_dataset.map(tf_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch datasets
    train_dataset = (
        train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
