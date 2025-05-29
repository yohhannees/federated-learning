import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime
import os

from utils.logging import setup_logging


class SecureAggregator:
    """Handles secure aggregation of client model weights."""

    def __init__(
        self,
        noise_scale: float = 0.1,
        clip_norm: float = 1.0,
        use_dp: bool = False,
        use_he: bool = False,
        seed: int = 42,
    ):
        """
        Initialize the secure aggregator.

        Args:
            noise_scale: Scale of noise for differential privacy
            clip_norm: Maximum L2 norm for gradient clipping
            use_dp: Whether to use differential privacy
            use_he: Whether to use homomorphic encryption
            seed: Random seed for reproducibility
        """
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.use_dp = use_dp
        self.use_he = use_he

        # Setup logging
        self.logger = setup_logging(
            "secure_aggregator",
            os.path.join(
                "logs",
                f'secure_aggregator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            ),
        )

        # Set random seed
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.logger.info(f"Initialized secure aggregator with DP={use_dp}, HE={use_he}")

    def aggregate(self, client_weights: List[List[tf.Tensor]]) -> List[tf.Tensor]:
        """
        Aggregate client weights using secure aggregation.

        Args:
            client_weights: List of client model weights

        Returns:
            Aggregated model weights
        """
        if not client_weights:
            self.logger.warning("No client weights to aggregate")
            return None

        try:
            # Convert weights to numpy arrays and ensure consistent shapes
            weights_array = []
            for client_weight in client_weights:
                client_weights_np = [
                    w.numpy() if isinstance(w, tf.Tensor) else w for w in client_weight
                ]
                weights_array.append(client_weights_np)

            # Apply differential privacy if enabled
            if self.use_dp:
                weights_array = self._apply_differential_privacy(weights_array)

            # Apply homomorphic encryption if enabled
            if self.use_he:
                weights_array = self._apply_homomorphic_encryption(weights_array)

            # Apply gradient clipping
            weights_array = self._clip_gradients(weights_array)

            # Aggregate weights using weighted average
            aggregated_weights = self._weighted_average(weights_array)

            self.logger.info("Successfully aggregated client weights")
            return [tf.convert_to_tensor(w) for w in aggregated_weights]

        except Exception as e:
            self.logger.error(f"Error in weight aggregation: {str(e)}")
            raise

    def _apply_differential_privacy(
        self, weights: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """Apply differential privacy by adding noise to weights."""
        noisy_weights = []
        for client_weights in weights:
            client_noisy_weights = []
            for w in client_weights:
                noise = np.random.normal(0, self.noise_scale, size=w.shape)
                client_noisy_weights.append(w + noise)
            noisy_weights.append(client_noisy_weights)
        return noisy_weights

    def _apply_homomorphic_encryption(
        self, weights: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """Apply homomorphic encryption to weights."""
        # Note: This is a simplified simulation of homomorphic encryption
        # In a real implementation, you would use a proper HE library
        return weights

    def _clip_gradients(
        self, weights: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """Clip gradients to prevent large updates."""
        clipped_weights = []
        for client_weights in weights:
            client_clipped_weights = []
            for w in client_weights:
                norm = np.linalg.norm(w)
                if norm > self.clip_norm:
                    w = w * (self.clip_norm / norm)
                client_clipped_weights.append(w)
            clipped_weights.append(client_clipped_weights)
        return clipped_weights

    def _weighted_average(self, weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Compute weighted average of weights."""
        # For simplicity, we use equal weights
        # In a real implementation, you might want to weight by dataset size
        num_layers = len(weights[0])
        aggregated_weights = []

        for layer_idx in range(num_layers):
            layer_weights = [client_weights[layer_idx] for client_weights in weights]
            aggregated_layer = np.mean(layer_weights, axis=0)
            aggregated_weights.append(aggregated_layer)

        return aggregated_weights

    def _detect_byzantine_faults(
        self, weights: List[List[np.ndarray]], threshold: float = 0.3
    ) -> List[List[np.ndarray]]:
        """
        Detect and remove Byzantine clients.

        Args:
            weights: List of client weights
            threshold: Threshold for Byzantine detection

        Returns:
            Filtered list of weights
        """
        if len(weights) < 3:  # Need at least 3 clients for detection
            return weights

        # Calculate distances for each layer
        distances = []
        for layer_idx in range(len(weights[0])):
            layer_weights = [client_weights[layer_idx] for client_weights in weights]
            median_weights = np.median(layer_weights, axis=0)
            layer_distances = np.array(
                [np.linalg.norm(w - median_weights) for w in layer_weights]
            )
            distances.append(layer_distances)

        # Average distances across layers
        avg_distances = np.mean(distances, axis=0)

        # Remove clients with large deviations
        threshold_value = np.percentile(avg_distances, (1 - threshold) * 100)
        valid_indices = avg_distances <= threshold_value

        if np.sum(valid_indices) < len(weights):
            self.logger.warning(
                f"Detected {len(weights) - np.sum(valid_indices)} Byzantine clients"
            )

        return [w for w, is_valid in zip(weights, valid_indices) if is_valid]
