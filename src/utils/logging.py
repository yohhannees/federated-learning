import logging
import os
from typing import Optional
import json
from datetime import datetime


def setup_logging(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp() -> str:
    """
    Get current timestamp in a formatted string.

    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class MetricsLogger:
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to store log files
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or get_timestamp()
        self.metrics_file = os.path.join(
            log_dir, f"{self.experiment_name}_metrics.json"
        )
        self.metrics = []

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

    def log_metrics(
        self,
        round_idx: int,
        global_metrics: dict,
        client_metrics: list,
        byzantine_flags: list,
    ):
        """
        Log training metrics for a round.

        Args:
            round_idx: Current round index
            global_metrics: Global model metrics
            client_metrics: List of client metrics
            byzantine_flags: List of Byzantine client flags
        """
        metrics = {
            "round": round_idx,
            "timestamp": get_timestamp(),
            "global_metrics": global_metrics,
            "client_metrics": client_metrics,
            "byzantine_flags": byzantine_flags,
        }

        self.metrics.append(metrics)

        # Save to file
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_latest_metrics(self) -> dict:
        """
        Get metrics from the latest round.

        Returns:
            Latest metrics dictionary
        """
        return self.metrics[-1] if self.metrics else None

    def get_metrics_history(self) -> list:
        """
        Get complete metrics history.

        Returns:
            List of all metrics dictionaries
        """
        return self.metrics

    def export_metrics(self, format: str = "json"):
        """
        Export metrics in specified format.

        Args:
            format: Export format ('json' or 'csv')
        """
        if format == "json":
            return self.metrics
        elif format == "csv":
            import pandas as pd

            # Flatten metrics for CSV
            flat_metrics = []
            for m in self.metrics:
                flat_metric = {
                    "round": m["round"],
                    "timestamp": m["timestamp"],
                    "global_loss": m["global_metrics"]["loss"],
                    "global_accuracy": m["global_metrics"]["accuracy"],
                }

                # Add client metrics
                for i, client_metric in enumerate(m["client_metrics"]):
                    flat_metric[f"client_{i}_accuracy"] = client_metric["accuracy"]
                    flat_metric[f"client_{i}_byzantine"] = m["byzantine_flags"][i]

                flat_metrics.append(flat_metric)

            return pd.DataFrame(flat_metrics)
        else:
            raise ValueError(f"Unsupported format: {format}")
