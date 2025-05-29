import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import os
import time

from utils.logging import setup_logging


class FederatedClient:
    """Represents a client in the federated learning system."""

    def __init__(
        self,
        client_id: int,
        model_fn: callable,
        dataset: tf.data.Dataset,
        learning_rate: float = 0.01,
        local_epochs: int = 5,
        batch_size: int = 32,
        seed: int = 42,
    ):
        """
        Initialize the federated client.

        Args:
            client_id: Unique identifier for the client
            model_fn: Function to create the model
            dataset: Client's local dataset
            learning_rate: Learning rate for local training
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            seed: Random seed for reproducibility
        """
        self.client_id = client_id
        self.model_fn = model_fn
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # Setup logging
        self.logger = setup_logging(
            f"client_{client_id}",
            os.path.join(
                "logs",
                f'client_{client_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            ),
        )

        # Set random seeds
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Initialize model and optimizer
        self.model = model_fn()
        self.optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)

        # Initialize metrics
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.metrics = {
            "loss": tf.keras.metrics.Mean(),
            "accuracy": tf.keras.metrics.SparseCategoricalAccuracy(),
        }

        self.logger.info(f"Initialized client {client_id}")

    def train_step(self, batch: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Perform a single training step.

        Args:
            batch: Tuple of (features, labels)

        Returns:
            Dictionary of metrics for the step
        """
        features, labels = batch

        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)
            loss = self.loss_fn(labels, predictions)

        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metrics
        self.metrics["loss"].update_state(loss)
        self.metrics["accuracy"].update_state(labels, predictions)

        return {"loss": loss, "accuracy": self.metrics["accuracy"].result()}

    def train(self, global_weights: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Train the client's model.

        Args:
            global_weights: Global model weights to initialize from

        Returns:
            Dictionary containing training metrics and model weights
        """
        try:
            start_time = time.time()

            # Initialize from global weights if provided
            if global_weights is not None:
                self.model.set_weights(global_weights)

            # Reset metrics
            for metric in self.metrics.values():
                metric.reset_states()

            # Training loop
            for epoch in range(self.local_epochs):
                epoch_start = time.time()

                for batch in self.dataset:
                    self.train_step(batch)

                epoch_time = time.time() - epoch_start
                self.logger.info(
                    f"Client {self.client_id} - Epoch {epoch + 1}/{self.local_epochs} - "
                    f"Loss: {self.metrics['loss'].result():.4f} - "
                    f"Accuracy: {self.metrics['accuracy'].result():.4f} - "
                    f"Time: {epoch_time:.2f}s"
                )

            training_time = time.time() - start_time

            # Get final metrics
            metrics = {
                "loss": float(self.metrics["loss"].result()),
                "accuracy": float(self.metrics["accuracy"].result()),
                "training_time": training_time,
            }

            # Get model weights
            weights = self.model.get_weights()

            self.logger.info(
                f"Client {self.client_id} completed training - "
                f"Final Loss: {metrics['loss']:.4f} - "
                f"Final Accuracy: {metrics['accuracy']:.4f} - "
                f"Total Time: {training_time:.2f}s"
            )

            return {"metrics": metrics, "weights": weights}

        except Exception as e:
            self.logger.error(f"Training failed for client {self.client_id}: {str(e)}")
            raise

    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the client's model on test data.

        Args:
            test_data: Test dataset

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Reset metrics
            for metric in self.metrics.values():
                metric.reset_states()

            # Evaluation loop
            for batch in test_data:
                features, labels = batch
                predictions = self.model(features, training=False)

                # Update metrics
                self.metrics["loss"].update_state(self.loss_fn(labels, predictions))
                self.metrics["accuracy"].update_state(labels, predictions)

            metrics = {
                "loss": float(self.metrics["loss"].result()),
                "accuracy": float(self.metrics["accuracy"].result()),
            }

            self.logger.info(
                f"Client {self.client_id} evaluation - "
                f"Loss: {metrics['loss']:.4f} - "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )

            return metrics

        except Exception as e:
            self.logger.error(
                f"Evaluation failed for client {self.client_id}: {str(e)}"
            )
            raise
