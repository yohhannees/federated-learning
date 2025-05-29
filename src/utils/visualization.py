import matplotlib.pyplot as plt
import os
import logging
from src.utils.logging import setup_logger
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import pandas as pd
import json
from datetime import datetime
from flask import Flask, render_template, jsonify
import threading
import queue
import plotly.express as px
import time

logger = setup_logger("visualization", "logs/visualization.log")


def plot_metrics(metrics_history):
    """Plot accuracy and loss over rounds."""
    logger.info("Generating metrics plots")
    try:
        rounds = range(1, len(metrics_history) + 1)
        accuracies = [m["sparse_categorical_accuracy"] for m in metrics_history]
        losses = [m["loss"] for m in metrics_history]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rounds, accuracies, label="Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Over Rounds")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(rounds, losses, label="Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Model Loss Over Rounds")
        plt.legend()

        os.makedirs("logs", exist_ok=True)
        plt.savefig("logs/metrics_plot.png")
        plt.close()
        logger.info("Metrics plot saved to logs/metrics_plot.png")
    except Exception as e:
        logger.error(f"Plotting failed: {str(e)}")
        raise


class TrainingVisualizer:
    def __init__(self, save_dir: str = "dashboards"):
        """
        Initialize the training visualizer.

        Args:
            save_dir: Directory to save visualization files
        """
        self.save_dir = save_dir
        self.history = {
            "loss": [],
            "accuracy": [],
            "client_metrics": [],
            "byzantine_flags": [],
        }

    def update_history(
        self,
        metrics: Dict[str, float],
        client_metrics: List[Dict[str, float]],
        byzantine_flags: List[bool],
    ):
        """
        Update training history with new metrics.

        Args:
            metrics: Global model metrics
            client_metrics: List of client metrics
            byzantine_flags: List of Byzantine client flags
        """
        self.history["loss"].append(metrics["loss"])
        self.history["accuracy"].append(metrics["accuracy"])
        self.history["client_metrics"].append(client_metrics)
        self.history["byzantine_flags"].append(byzantine_flags)

    def plot_training_progress(self, save_path: str = None):
        """
        Plot training progress using matplotlib.

        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history["loss"], label="Training Loss")
        plt.title("Training Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history["accuracy"], label="Training Accuracy")
        plt.title("Training Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved training progress plot to {save_path}")
        else:
            plt.show()

    def create_interactive_dashboard(self, save_path: str = None):
        """
        Create an interactive dashboard using Plotly.

        Args:
            save_path: Path to save the HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Training Loss",
                "Training Accuracy",
                "Client Performance",
                "Byzantine Detection",
            ),
        )

        # Add loss trace
        fig.add_trace(
            go.Scatter(y=self.history["loss"], name="Loss", line=dict(color="blue")),
            row=1,
            col=1,
        )

        # Add accuracy trace
        fig.add_trace(
            go.Scatter(
                y=self.history["accuracy"], name="Accuracy", line=dict(color="green")
            ),
            row=1,
            col=2,
        )

        # Add client performance heatmap
        client_metrics = np.array(
            [
                [m["accuracy"] for m in round_metrics]
                for round_metrics in self.history["client_metrics"]
            ]
        )

        fig.add_trace(
            go.Heatmap(z=client_metrics, colorscale="Viridis", name="Client Accuracy"),
            row=2,
            col=1,
        )

        # Add Byzantine detection heatmap
        byzantine_flags = np.array(self.history["byzantine_flags"])
        fig.add_trace(
            go.Heatmap(z=byzantine_flags, colorscale="RdBu", name="Byzantine Clients"),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title="Federated Learning Dashboard", height=800, showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive dashboard to {save_path}")
        else:
            fig.show()

    def plot_client_distribution(self, client_metrics: List[Dict[str, float]]):
        """
        Plot distribution of client performance metrics.

        Args:
            client_metrics: List of client metrics
        """
        # Extract accuracies
        accuracies = [m["accuracy"] for m in client_metrics]

        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(accuracies, bins=20, alpha=0.7)
        plt.title("Distribution of Client Accuracies")
        plt.xlabel("Accuracy")
        plt.ylabel("Number of Clients")
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        plt.axvline(
            mean_acc, color="red", linestyle="--", label=f"Mean: {mean_acc:.3f}"
        )
        plt.axvline(
            mean_acc + std_acc,
            color="green",
            linestyle=":",
            label=f"Std: {std_acc:.3f}",
        )
        plt.axvline(mean_acc - std_acc, color="green", linestyle=":")

        plt.legend()
        plt.show()

    def export_metrics(self, save_path: str):
        """
        Export training metrics to CSV.

        Args:
            save_path: Path to save the CSV file
        """
        # Create DataFrame
        df = pd.DataFrame(
            {
                "round": range(len(self.history["loss"])),
                "loss": self.history["loss"],
                "accuracy": self.history["accuracy"],
            }
        )

        # Add client metrics
        for i, round_metrics in enumerate(self.history["client_metrics"]):
            for j, client_metric in enumerate(round_metrics):
                df.loc[i, f"client_{j}_accuracy"] = client_metric["accuracy"]

        # Add Byzantine flags
        for i, round_flags in enumerate(self.history["byzantine_flags"]):
            for j, is_byzantine in enumerate(round_flags):
                df.loc[i, f"client_{j}_byzantine"] = is_byzantine

        # Save to CSV
        df.to_csv(save_path, index=False)
        logger.info(f"Exported metrics to {save_path}")


class TrainingMonitor:
    """Monitors and visualizes federated learning training progress."""

    def __init__(
        self,
        log_dir: str = "logs",
        results_dir: str = "results",
        port: int = 5000,
    ):
        """
        Initialize the training monitor.

        Args:
            log_dir: Directory for log files
            results_dir: Directory for saving results and plots
            port: Port number for Flask server
        """
        self.log_dir = log_dir
        self.results_dir = results_dir
        self.port = port

        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(
            "training_monitor",
            os.path.join(
                log_dir,
                f'training_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            ),
        )

        # Initialize Flask app
        self.app = Flask(__name__)
        self.setup_routes()

        # Initialize metrics storage
        self.metrics = {
            "round": [],
            "global_loss": [],
            "global_accuracy": [],
            "client_metrics": {},
            "participation": [],
            "adversarial_impact": [],
        }

        # Queue for thread-safe updates
        self.update_queue = queue.Queue()

        # Start Flask server in a separate thread
        self.server_thread = threading.Thread(
            target=self.app.run,
            kwargs={"host": "0.0.0.0", "port": port, "debug": False},
        )
        self.server_thread.daemon = True
        self.server_thread.start()

        self.logger.info(f"Training monitor initialized on port {port}")

    def setup_routes(self):
        """Setup Flask routes for the dashboard."""

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/metrics")
        def get_metrics():
            return jsonify(self.metrics)

        @self.app.route("/plots")
        def get_plots():
            plots = self._generate_plots()
            return jsonify(plots)

    def update_metrics(
        self,
        round_num: int,
        global_metrics: dict,
        client_metrics: dict,
        participation: list,
        adversarial_impact: float = 0.0,
    ):
        """
        Update training metrics.

        Args:
            round_num: Current round number
            global_metrics: Global model metrics
            client_metrics: Per-client metrics
            participation: List of participating client IDs
            adversarial_impact: Impact of adversarial behavior
        """
        try:
            # Update metrics
            self.metrics["round"].append(round_num)
            self.metrics["global_loss"].append(global_metrics["loss"])
            self.metrics["global_accuracy"].append(global_metrics["accuracy"])
            self.metrics["participation"].append(participation)
            self.metrics["adversarial_impact"].append(adversarial_impact)

            # Update client metrics
            for client_id, metrics in client_metrics.items():
                if client_id not in self.metrics["client_metrics"]:
                    self.metrics["client_metrics"][client_id] = {
                        "loss": [],
                        "accuracy": [],
                    }
                self.metrics["client_metrics"][client_id]["loss"].append(
                    metrics["loss"]
                )
                self.metrics["client_metrics"][client_id]["accuracy"].append(
                    metrics["accuracy"]
                )

            # Save metrics to file
            self._save_metrics()

            # Generate and save plots
            self._generate_plots()

            self.logger.info(f"Updated metrics for round {round_num}")

        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            raise

    def _save_metrics(self):
        """Save metrics to JSON file."""
        try:
            # Convert metrics to serializable format
            serializable_metrics = {
                k: v if isinstance(v, (list, dict)) else float(v)
                for k, v in self.metrics.items()
            }

            # Save to file
            with open(
                os.path.join(self.results_dir, "training_metrics.json"), "w"
            ) as f:
                json.dump(serializable_metrics, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise

    def _generate_plots(self) -> dict:
        """
        Generate training progress plots.

        Returns:
            Dictionary of plot data
        """
        try:
            plots = {}

            # Global metrics plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.metrics["round"],
                    y=self.metrics["global_loss"],
                    name="Global Loss",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.metrics["round"],
                    y=self.metrics["global_accuracy"],
                    name="Global Accuracy",
                )
            )
            fig.update_layout(
                title="Global Model Performance",
                xaxis_title="Round",
                yaxis_title="Metric Value",
            )
            plots["global_metrics"] = fig.to_json()

            # Client participation plot
            participation_data = []
            for round_num, clients in enumerate(self.metrics["participation"]):
                for client_id in clients:
                    participation_data.append({"round": round_num, "client": client_id})
            fig = px.scatter(
                participation_data,
                x="round",
                y="client",
                title="Client Participation",
            )
            plots["participation"] = fig.to_json()

            # Adversarial impact plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.metrics["round"],
                    y=self.metrics["adversarial_impact"],
                    name="Adversarial Impact",
                )
            )
            fig.update_layout(
                title="Impact of Adversarial Behavior",
                xaxis_title="Round",
                yaxis_title="Impact Score",
            )
            plots["adversarial_impact"] = fig.to_json()

            # Save plots
            for name, plot_data in plots.items():
                with open(os.path.join(self.results_dir, f"{name}.json"), "w") as f:
                    f.write(plot_data)

            return plots

        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            raise

    def create_html_template(self):
        """Create HTML template for the dashboard."""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Federated Learning Monitor</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .plot-container { margin: 20px 0; }
                .metrics-container { display: flex; flex-wrap: wrap; }
                .metric-card {
                    background: #f5f5f5;
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                    min-width: 200px;
                }
            </style>
        </head>
        <body>
            <h1>Federated Learning Training Monitor</h1>
            
            <div class="metrics-container">
                <div class="metric-card">
                    <h3>Global Loss</h3>
                    <div id="global-loss"></div>
                </div>
                <div class="metric-card">
                    <h3>Global Accuracy</h3>
                    <div id="global-accuracy"></div>
                </div>
            </div>

            <div class="plot-container">
                <h2>Training Progress</h2>
                <div id="training-progress"></div>
            </div>

            <div class="plot-container">
                <h2>Client Participation</h2>
                <div id="client-participation"></div>
            </div>

            <div class="plot-container">
                <h2>Adversarial Impact</h2>
                <div id="adversarial-impact"></div>
            </div>

            <script>
                function updatePlots() {
                    fetch('/plots')
                        .then(response => response.json())
                        .then(data => {
                            Plotly.newPlot('training-progress', JSON.parse(data.global_metrics));
                            Plotly.newPlot('client-participation', JSON.parse(data.participation));
                            Plotly.newPlot('adversarial-impact', JSON.parse(data.adversarial_impact));
                        });
                }

                function updateMetrics() {
                    fetch('/metrics')
                        .then(response => response.json())
                        .then(data => {
                            const latest = data.round.length - 1;
                            document.getElementById('global-loss').textContent = 
                                data.global_loss[latest].toFixed(4);
                            document.getElementById('global-accuracy').textContent = 
                                data.global_accuracy[latest].toFixed(4);
                        });
                }

                // Update every 5 seconds
                setInterval(() => {
                    updatePlots();
                    updateMetrics();
                }, 5000);

                // Initial update
                updatePlots();
                updateMetrics();
            </script>
        </body>
        </html>
        """

        # Create templates directory if it doesn't exist
        os.makedirs("templates", exist_ok=True)

        # Save template
        with open("templates/index.html", "w") as f:
            f.write(template)

        self.logger.info("Created HTML template for dashboard")
