import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import os
import time

from utils.logging import setup_logging
from client.client import FederatedClient

logger = logging.getLogger(__name__)


class AdversarialClient(FederatedClient):
    """Represents an adversarial client in the federated learning system."""

    def __init__(
        self,
        client_id: int,
        model_fn: callable,
        dataset: tf.data.Dataset,
        attack_type: str = "label_flip",
        noise_scale: float = 0.1,
        compute_delay: float = 0.0,
        learning_rate: float = 0.01,
        local_epochs: int = 5,
        batch_size: int = 32,
        seed: int = 42,
    ):
        """
        Initialize the adversarial client.

        Args:
            client_id: Unique identifier for the client
            model_fn: Function to create the model
            dataset: Client's local dataset
            attack_type: Type of adversarial attack ('label_flip' or 'noise')
            noise_scale: Scale of noise to add for noise-based attacks
            compute_delay: Simulated computation delay in seconds
            learning_rate: Learning rate for local training
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            seed: Random seed for reproducibility
        """
        super().__init__(
            client_id=client_id,
            model_fn=model_fn,
            dataset=dataset,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
        )

        self.attack_type = attack_type
        self.noise_scale = noise_scale
        self.compute_delay = compute_delay

        # Setup logging
        self.logger = setup_logging(
            f"adversarial_client_{client_id}",
            os.path.join(
                "logs",
                f'adversarial_client_{client_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            ),
        )

        self.logger.info(
            f"Initialized adversarial client {client_id} with attack type: {attack_type}"
        )

    def train(self, global_weights: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Train the client's model with adversarial behavior.

        Args:
            global_weights: Global model weights to initialize from

        Returns:
            Dictionary containing training metrics and model weights
        """
        try:
            # Simulate computation delay
            if self.compute_delay > 0:
                time.sleep(self.compute_delay)

            # Train model normally first
            results = super().train(global_weights)

            # Apply adversarial behavior to weights
            results["weights"] = self._apply_adversarial_behavior(results["weights"])

            self.logger.info(
                f"Applied {self.attack_type} attack to client {self.client_id}"
            )

            return results

        except Exception as e:
            self.logger.error(
                f"Adversarial training failed for client {self.client_id}: {str(e)}"
            )
            raise

    def _apply_adversarial_behavior(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Apply adversarial behavior to model weights.

        Args:
            weights: Original model weights

        Returns:
            Modified weights
        """
        if self.attack_type == "label_flip":
            # Label flipping attack: flip signs of weights
            return [-w for w in weights]

        elif self.attack_type == "noise":
            # Noise injection attack: add random noise to weights
            noisy_weights = []

            for w in weights:
                noise = tf.random.normal(w.shape, mean=0.0, stddev=self.noise_scale)
                noisy_weights.append(w + noise)

            return noisy_weights

        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def _apply_label_flip_attack(
        self, weights: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """
        Apply label flipping attack by modifying the final layer weights.

        Args:
            weights: Original model weights

        Returns:
            Modified weights
        """
        # Get the final layer weights
        final_layer_weights = weights[-2]  # Dense layer weights
        final_layer_bias = weights[-1]  # Dense layer bias

        # Create a new weight matrix with flipped labels
        new_weights = np.zeros_like(final_layer_weights)
        new_bias = np.zeros_like(final_layer_bias)

        # Apply label flipping
        for old_label, new_label in self.label_map.items():
            new_weights[:, new_label] = final_layer_weights[:, old_label]
            new_bias[new_label] = final_layer_bias[old_label]

        # Create new weights list
        new_weights_list = weights.copy()
        new_weights_list[-2] = new_weights
        new_weights_list[-1] = new_bias

        logger.info(f"Applied label flipping attack to client {self.client_id}")
        return new_weights_list

    def _apply_noise_attack(
        self, weights: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """
        Apply noise attack by adding Gaussian noise to weights.

        Args:
            weights: Original model weights

        Returns:
            Modified weights
        """
        noisy_weights = []

        for w in weights:
            # Calculate noise scale based on weight statistics
            weight_std = np.std(w)
            noise_scale = weight_std * self.attack_strength

            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, w.shape)
            noisy_weights.append(w + noise)

        logger.info(f"Applied noise attack to client {self.client_id}")
        return noisy_weights

    def train_step(self, batch: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Perform a single training step with adversarial behavior.

        Args:
            batch: Tuple of (features, labels)

        Returns:
            Dictionary of metrics for this step
        """
        features, labels = batch

        # Apply label flipping to training data if using label flip attack
        if self.attack_type == "label_flip":
            # Flip labels (0->9, 1->8, 2->7, etc.)
            labels = 9 - labels

        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)
            loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.metrics["loss"].update_state(loss)
        self.metrics["accuracy"].update_state(labels, predictions)

        return {"loss": loss, "accuracy": self.metrics["accuracy"].result()}
