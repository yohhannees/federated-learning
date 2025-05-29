import tensorflow as tf
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from scipy import stats
from src.utils.logging import setup_logger

logger = setup_logger("secure_aggregation", "logs/secure_aggregation.log")


def detect_byzantine_weights(
    weights: List[np.ndarray], threshold: float = 2.0
) -> List[bool]:
    """
    Detect Byzantine clients using median absolute deviation.

    Args:
        weights: List of client weight updates
        threshold: Threshold for outlier detection

    Returns:
        List of boolean flags indicating Byzantine clients
    """
    # Stack weights into a single array
    stacked_weights = np.stack(weights)

    # Calculate median and MAD for each weight
    median = np.median(stacked_weights, axis=0)
    mad = stats.median_abs_deviation(stacked_weights, axis=0)

    # Detect outliers
    is_byzantine = []
    for client_weights in weights:
        deviation = np.abs(client_weights - median)
        is_outlier = np.any(deviation > (threshold * mad))
        is_byzantine.append(is_outlier)

    return is_byzantine


def secure_aggregate(
    client_weights: List[Dict[str, tf.Tensor]],
    client_metrics: List[Dict[str, float]],
    byzantine_threshold: float = 2.0,
) -> Tuple[Dict[str, tf.Tensor], List[bool]]:
    """
    Perform secure aggregation with Byzantine fault tolerance.

    Args:
        client_weights: List of client model weights
        client_metrics: List of client training metrics
        byzantine_threshold: Threshold for Byzantine detection

    Returns:
        Tuple of (aggregated weights, byzantine flags)
    """
    # Convert weights to numpy arrays for easier manipulation
    numpy_weights = []
    for weights in client_weights:
        numpy_weights.append([w.numpy() for w in weights])

    # Detect Byzantine clients
    byzantine_flags = []
    for layer_idx in range(len(numpy_weights[0])):
        layer_weights = [w[layer_idx] for w in numpy_weights]
        layer_byzantine = detect_byzantine_weights(
            layer_weights, threshold=byzantine_threshold
        )
        byzantine_flags.append(layer_byzantine)

    # Combine Byzantine flags across layers
    is_byzantine = np.any(byzantine_flags, axis=0)

    # Filter out Byzantine clients
    valid_weights = []
    valid_metrics = []
    for idx, (weights, metrics) in enumerate(zip(client_weights, client_metrics)):
        if not is_byzantine[idx]:
            valid_weights.append(weights)
            valid_metrics.append(metrics)

    if not valid_weights:
        logger.warning("All clients detected as Byzantine!")
        return client_weights[0], is_byzantine

    # Calculate weighted average based on client metrics
    weights = []
    total_weight = 0.0

    for client_metric in valid_metrics:
        # Use accuracy as weight
        weight = client_metric["accuracy"]
        weights.append(weight)
        total_weight += weight

    # Normalize weights
    weights = [w / total_weight for w in weights]

    # Aggregate weights
    aggregated_weights = []
    for layer_idx in range(len(valid_weights[0])):
        layer_weights = np.zeros_like(valid_weights[0][layer_idx])
        for client_idx, client_weight in enumerate(valid_weights):
            layer_weights += weights[client_idx] * client_weight[layer_idx]
        aggregated_weights.append(tf.convert_to_tensor(layer_weights))

    logger.info(
        f"Secure aggregation completed - "
        f"Byzantine clients: {np.sum(is_byzantine)}/{len(is_byzantine)}"
    )
    return aggregated_weights, is_byzantine


def add_differential_privacy(
    weights: Dict[str, tf.Tensor], noise_scale: float = 0.1
) -> Dict[str, tf.Tensor]:
    """
    Add differential privacy noise to weights.

    Args:
        weights: Model weights
        noise_scale: Scale of Gaussian noise

    Returns:
        Noisy weights
    """
    noisy_weights = []

    for w in weights:
        noise = tf.random.normal(w.shape, mean=0.0, stddev=noise_scale)
        noisy_weights.append(w + noise)

    return noisy_weights
