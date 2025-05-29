import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from datetime import datetime
import os

from utils.logging import setup_logging


class DataPartitioner:
    """Handles data partitioning for federated learning."""

    def __init__(
        self,
        dataset_name: str = "mnist",
        num_clients: int = 10,
        alpha: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize the data partitioner.

        Args:
            dataset_name: Name of the dataset to use
            num_clients: Number of clients to partition data for
            alpha: Concentration parameter for Dirichlet distribution
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.alpha = alpha

        # Setup logging
        self.logger = setup_logging(
            "data_partitioner",
            os.path.join(
                "logs",
                f'data_partitioner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            ),
        )

        # Set random seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.logger.info(
            f"Initialized data partitioner for {dataset_name} with {num_clients} clients"
        )

    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load the specified dataset.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if self.dataset_name == "mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

            # Normalize pixel values
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0

            # Create TensorFlow datasets
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

            self.logger.info("Loaded MNIST dataset")
            return train_dataset, test_dataset
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def create_non_iid_partition(
        self, dataset: tf.data.Dataset, batch_size: int = 32
    ) -> List[tf.data.Dataset]:
        """
        Create non-IID partitions using Dirichlet distribution.

        Args:
            dataset: Input dataset to partition
            batch_size: Batch size for the datasets

        Returns:
            List of client datasets
        """
        # Convert dataset to numpy arrays
        x_data = []
        y_data = []
        for x, y in dataset:
            x_data.append(x.numpy())
            y_data.append(y.numpy())
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Get unique classes
        num_classes = len(np.unique(y_data))
        client_data = [[] for _ in range(self.num_clients)]

        # Create non-IID distribution using Dirichlet
        for k in range(num_classes):
            # Get indices for current class
            idx_k = np.where(y_data == k)[0]
            np.random.shuffle(idx_k)

            # Create proportions using Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))

            # Calculate number of samples per client
            proportions = np.array([p * len(idx_k) for p in proportions])
            proportions = proportions.astype(int)

            # Adjust for rounding errors
            proportions[-1] = len(idx_k) - np.sum(proportions[:-1])

            # Distribute data to clients
            start_idx = 0
            for i in range(self.num_clients):
                end_idx = start_idx + proportions[i]
                client_data[i].extend(idx_k[start_idx:end_idx])
                start_idx = end_idx

        # Create client datasets
        client_datasets = []
        for i in range(self.num_clients):
            indices = client_data[i]
            client_x = x_data[indices]
            client_y = y_data[indices]

            # Create TensorFlow dataset
            client_dataset = tf.data.Dataset.from_tensor_slices((client_x, client_y))
            client_dataset = client_dataset.shuffle(len(indices)).batch(batch_size)
            client_datasets.append(client_dataset)

        self.logger.info(f"Created non-IID partitions for {self.num_clients} clients")
        return client_datasets

    def get_client_data(
        self, batch_size: int = 32
    ) -> Tuple[List[tf.data.Dataset], tf.data.Dataset]:
        """
        Get partitioned data for clients.

        Args:
            batch_size: Batch size for the datasets

        Returns:
            Tuple of (client_datasets, test_dataset)
        """
        # Load dataset
        train_dataset, test_dataset = self.load_dataset()

        # Create non-IID partitions
        client_datasets = self.create_non_iid_partition(train_dataset, batch_size)

        # Prepare test dataset
        test_dataset = test_dataset.batch(batch_size)

        self.logger.info(
            f"Prepared data for {len(client_datasets)} clients with batch size {batch_size}"
        )
        return client_datasets, test_dataset

    def get_data_stats(self, client_datasets: List[tf.data.Dataset]) -> Dict[str, Any]:
        """
        Get statistics about the data distribution.

        Args:
            client_datasets: List of client datasets

        Returns:
            Dictionary containing data statistics
        """
        stats = {
            "total_clients": len(client_datasets),
            "client_sizes": [],
            "class_distribution": [],
        }

        for i, dataset in enumerate(client_datasets):
            # Count samples per class
            class_counts = np.zeros(10)  # For MNIST
            total_samples = 0

            for _, labels in dataset:
                for label in labels:
                    class_counts[label.numpy()] += 1
                    total_samples += 1

            stats["client_sizes"].append(total_samples)
            stats["class_distribution"].append(class_counts.tolist())

        self.logger.info("Generated data distribution statistics")
        return stats
