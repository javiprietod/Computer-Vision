# deep leanring libraries
import jax
from torchvision import transforms
import numpy as np
import jax.numpy as jnp

# other libraries
import os
import math
import requests
import tarfile
import shutil
from PIL import Image
from typing import Iterator
from requests.models import Response
from tarfile import TarFile


class JaxDataset:
    def __init__(self, path: str) -> None:
        """
        Constructor of Dataset.

        Args:
            path: path of the dataset.
        """

        # set attributes
        self.path = path
        self.names = sorted(os.listdir(path))

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """

        return len(self.names)

    def __getitem__(self, index: int) -> tuple[jax.Array, int]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with image and label. Image dimensions:
                [height, width, channels].
        """

        # TODO
        filename = self.names[index]
        image_np = np.load(f"{self.path}/{filename}")
        image_jax = jnp.array(image_np)
        # Return the tensor and the label
        return image_jax, int(filename.split("_")[0])


class JaxDataLoader:
    """
    _summary_
    """

    dataset: JaxDataset
    batch_size: int
    shuffle: bool
    drop_last: bool
    length: int
    batches_indexes: jax.Array

    def __init__(
        self,
        dataset: JaxDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        key: jax.Array = jax.random.key(0),
    ) -> None:
        """
        Constructor of Jax DataLoader.

        Args:
            path: path of the dataset.
        """

        # set attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.current_batch_idx = 0

        # get indexes
        self.indexes = self.get_indexes(shuffle, key)

    def get_indexes(self, shuffle: bool, key: jax.Array) -> jax.Array:
        """
        This function computes the indexes.

        Args:
            shuffle: indicator to shuffle the elements.
            key: key to compute random computations.

        Returns:
            indexes shuffled.
        """

        # TODO
        indexes = jnp.arange(len(self.dataset))
        if shuffle:
            indexes = jax.random.permutation(key, indexes)
        return indexes

    def __iter__(self) -> Iterator:
        """
        This method return the iterator we are building.

        Returns:
            iterator we are using.
        """

        # TODO
        self.current_batch_idx = 0
        return self

    def __len__(self) -> int:
        """
        This method computes the length of the dataset.

        Returns:
            length of the dataset.
        """

        return math.ceil(len(self.indexes) / self.batch_size)

    def __next__(self) -> tuple[jax.Array, jax.Array]:
        """
        This method computes the next element for each iteration of
        the iterator object.

        Raises:
            StopIteration: raised when the iterator ends.

        Returns:
            images array. Dimensions: [batch, height, width, channels].
            labels array. Dimensions: [batch].
        """

        # TODO
        if self.current_batch_idx >= len(self):
            raise StopIteration
        else:
            start = self.current_batch_idx * self.batch_size
            end = start + self.batch_size
            if self.drop_last and end > len(self.indexes):
                raise StopIteration
            indexes = self.indexes[start:end]
            images, labels = [], []
            for idx in indexes:
                image, label = self.dataset[idx]
                images.append(image)
                labels.append(label)
            self.current_batch_idx += 1
            return jnp.stack(images), jnp.array(labels)


def random_jax_split(
    dataloader: JaxDataLoader,
    splits: tuple[float, ...],
    key: jax.Array = jax.random.key(0),
) -> tuple[JaxDataLoader, ...]:
    """
    This is the function to divide a JaxDataLoader into splits.

    Args:
        dataloader: dataloader of jax.
        splits: splits with proportions. They have to sum 1.
        key: key for random number generation in jax.
            Defaults to jax.random.key(0).

    Returns:
        tuple of JaxDataLoaders. As many, as elements the splits have.
    """

    # TODO
    # Check if the splits sum 1
    assert sum(splits) == 1, "The splits do not sum 1"
    # Get the indexes
    # Get the full dataset from the dataloader
    dataset_size = len(dataloader.indexes)

    # Calculate number of elements for each split
    split_sizes = (np.array(splits) * dataset_size).astype(int)

    # Adjust any rounding issues
    split_sizes[-1] = dataset_size - np.sum(split_sizes[:-1])

    # Shuffle the indices
    indices = jnp.arange(dataset_size)
    key, subkey = jax.random.split(key)
    shuffled_indices = jax.random.permutation(subkey, indices)

    # Split the dataset indices based on the split sizes
    split_datasets = []
    start_idx = 0
    for split_size in split_sizes:
        end_idx = start_idx + split_size
        split_indices = shuffled_indices[start_idx:end_idx]

        # Create a new dataloader for the split
        split_dataloader = JaxDataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            drop_last=dataloader.drop_last,
            key=key,
        )
        split_dataloader.indexes = split_indices
        split_datasets.append(split_dataloader)

    return tuple(split_datasets)


def load_jax_data(
    path: str, batch_size: int = 128
) -> tuple[JaxDataLoader, JaxDataLoader, JaxDataLoader]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for imagenette dataset.

    Args:
        path: path of the dataset.
        batch_size: batch size for dataloaders. Default value: 128.

    Returns:
        tuple of dataloaders, train, val and test in respective order.
    """

    # TODO
    if not os.path.isdir(path):
        download_data(path)

    train_dataset: JaxDataset = JaxDataset(f"{path}/train")
    test_dataset: JaxDataset = JaxDataset(f"{path}/val")

    # Split the dataset into train, validation, and test
    key = jax.random.PRNGKey(0)

    train_loader, val_loader = random_jax_split(
        JaxDataLoader(dataset=train_dataset, batch_size=batch_size),
        splits=(0.7, 0.3),
        key=key,
    )
    test_loader = JaxDataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def download_data(path: str) -> None:
    """
    This function downloads the data from internet.

    Args:

    """
    # define paths
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
