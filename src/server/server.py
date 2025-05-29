import tensorflow as tf
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os

from utils.logging import setup_logging
from utils.secure_aggregation import SecureAggregator
from utils.communication import CommunicationProtocol


class FederatedServer:
    """Represents the central server in the federated learning system."""

    def __init__(
        self,
        model_fn: callable,
        secure_aggregator: SecureAggregator,
        communication: CommunicationProtocol,
        seed: int = 42,
    ):
        """
        Initialize the federated server.

        Args:
            model_fn: Function to create the model
            secure_aggregator: Secure aggregation utility
            communication: Communication protocol utility
            seed: Random seed for reproducibility
        """
        self.model_fn = model_fn
        self.secure_aggregator = secure_aggregator
        self.communication = communication

        # Setup logging
        self.logger = setup_logging(
            "server",
            os.path.join(
                "logs", f'server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
        )

        # Set random seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Initialize global model
        self.global_model = model_fn()
        self.global_weights = self.global_model.get_weights()

        self.logger.info("Initialized federated server")

    def select_clients(self, clients: List[Any], fraction: float = 0.1) -> List[Any]:
        """
        Select a subset of clients for training.

        Args:
            clients: List of all clients
            fraction: Fraction of clients to select

        Returns:
            List of selected clients
        """
        num_clients = max(1, int(len(clients) * fraction))
        selected_clients = np.random.choice(clients, num_clients, replace=False)

        self.logger.info(f"Selected {num_clients} clients for training")
        return selected_clients

    def aggregate_weights(
        self, client_weights: List[List[tf.Tensor]]
    ) -> List[tf.Tensor]:
        """
        Aggregate client weights using secure aggregation.

        Args:
            client_weights: List of client model weights

        Returns:
            Aggregated model weights
        """
        if not client_weights:
            return self.global_weights

        # Apply secure aggregation
        aggregated_weights = self.secure_aggregator.aggregate(client_weights)

        self.logger.info("Aggregated client weights")
        return aggregated_weights

    def update_global_weights(self, weights: List[tf.Tensor]):
        """
        Update the global model weights.

        Args:
            weights: New model weights
        """
        self.global_weights = weights
        self.global_model.set_weights(weights)
        self.logger.info("Updated global model weights")

    def get_global_weights(self) -> List[tf.Tensor]:
        """
        Get the current global model weights.

        Returns:
            Global model weights
        """
        return self.global_weights

    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the global model on test data.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "loss": tf.keras.metrics.Mean(),
            "accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
        }

        for batch in test_data:
            features, labels = batch
            predictions = self.global_model(features, training=False)

            # Update metrics
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            metrics["loss"].update_state(loss)
            metrics["accuracy"].update_state(labels, predictions)

        results = {
            "loss": float(metrics["loss"].result()),
            "accuracy": float(metrics["accuracy"].result()),
        }

        self.logger.info(
            f"Global model evaluation - Loss: {results['loss']:.4f} - Accuracy: {results['accuracy']:.4f}"
        )
        return results
