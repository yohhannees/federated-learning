import os
import logging
import argparse
import tensorflow as tf
from datetime import datetime

from data.data_partition import DataPartitioner
from client.client import FederatedClient
from client.adversarial_client import AdversarialClient
from server.server import FederatedServer
from utils.logging import setup_logging
from utils.secure_aggregation import SecureAggregator
from utils.communication import CommunicationProtocol


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")

    # Dataset parameters
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset to use (default: mnist)"
    )
    parser.add_argument(
        "--num-clients", type=int, default=10, help="Number of clients (default: 10)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter (default: 0.5)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of local epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of federated rounds (default: 100)",
    )

    # Client parameters
    parser.add_argument(
        "--adversarial-ratio",
        type=float,
        default=0.2,
        help="Ratio of adversarial clients (default: 0.2)",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        default="label_flip",
        choices=["label_flip", "noise_injection"],
        help="Type of adversarial attack (default: label_flip)",
    )

    # Security parameters
    parser.add_argument(
        "--use-dp", action="store_true", help="Use differential privacy"
    )
    parser.add_argument(
        "--use-he", action="store_true", help="Use homomorphic encryption"
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.1,
        help="Noise scale for differential privacy (default: 0.1)",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0)",
    )

    # Communication parameters
    parser.add_argument(
        "--async-mode", action="store_true", help="Use asynchronous communication"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Communication timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed communications (default: 3)",
    )

    return parser.parse_args()


def create_model():
    """Create and compile the model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    """Main simulation function."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging(
        "simulation",
        os.path.join(
            "logs", f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
    )

    logger.info("Starting federated learning simulation")
    logger.info(f"Arguments: {args}")

    # Initialize components
    data_partitioner = DataPartitioner(
        dataset_name=args.dataset, num_clients=args.num_clients, alpha=args.alpha
    )

    # Get client data
    client_datasets, test_dataset = data_partitioner.get_client_data(
        batch_size=args.batch_size
    )

    # Create secure aggregator
    secure_aggregator = SecureAggregator(
        noise_scale=args.noise_scale,
        clip_norm=args.clip_norm,
        use_dp=args.use_dp,
        use_he=args.use_he,
    )

    # Create communication protocol
    communication = CommunicationProtocol(
        timeout=args.timeout, max_retries=args.max_retries, async_mode=args.async_mode
    )

    # Create server
    server = FederatedServer(
        model_fn=create_model,
        secure_aggregator=secure_aggregator,
        communication=communication,
    )

    # Create clients
    clients = []
    num_adversarial = int(args.num_clients * args.adversarial_ratio)

    for i in range(args.num_clients):
        if i < num_adversarial:
            client = AdversarialClient(
                client_id=i,
                model_fn=create_model,
                dataset=client_datasets[i],
                attack_type=args.attack_type,
            )
        else:
            client = FederatedClient(
                client_id=i, model_fn=create_model, dataset=client_datasets[i]
            )
        clients.append(client)
        communication.register_client(i)

    logger.info(f"Created {len(clients)} clients ({num_adversarial} adversarial)")

    # Training loop
    for round_num in range(args.rounds):
        logger.info(f"Starting round {round_num + 1}/{args.rounds}")

        # Select clients for this round
        selected_clients = server.select_clients(clients)

        # Train selected clients
        client_weights = []
        for client in selected_clients:
            result = client.train(server.get_global_weights())
            client_weights.append(result["weights"])

        # Aggregate weights
        global_weights = server.aggregate_weights(client_weights)
        server.update_global_weights(global_weights)

        # Evaluate
        metrics = server.evaluate(test_dataset)
        logger.info(f"Round {round_num + 1} metrics: {metrics}")

    logger.info("Simulation completed")

    # Cleanup
    for client in clients:
        communication.unregister_client(client.client_id)


if __name__ == "__main__":
    main()
