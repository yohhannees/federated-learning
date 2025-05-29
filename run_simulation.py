import argparse
import pickle
import os
from src.server.server import federated_averaging
from src.utils.visualization import plot_metrics
from src.utils.logging import setup_logger

logger = setup_logger('simulation', 'logs/simulation.log')

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of training rounds')
    parser.add_argument('--clients_per_round', type=int, default=10, help='Clients per round')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Client dropout rate')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='Noise scale for secure aggregation')
    args = parser.parse_args()
    
    logger.info(f'Starting simulation: rounds={args.num_rounds}, clients={args.clients_per_round}, '
                f'dropout={args.dropout_rate}, noise={args.noise_scale}')
    
    try:
        state, metrics_history, test_dataset = federated_averaging(
            num_rounds=args.num_rounds,
            clients_per_round=args.clients_per_round,
            dropout_rate=args.dropout_rate,
            noise_scale=args.noise_scale
        )
        plot_metrics(metrics_history)
        
        # Save metrics for dashboard
        os.makedirs('logs', exist_ok=True)
        with open('logs/metrics_history.pkl', 'wb') as f:
            pickle.dump(metrics_history, f)
        logger.info('Metrics history saved to logs/metrics_history.pkl')
    except Exception as e:
        logger.error(f'Simulation failed: {str(e)}')
        raise

if __name__ == '__main__':
    main()