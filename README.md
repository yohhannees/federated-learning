# Federated Learning System

A robust implementation of a federated learning system with support for client heterogeneity, adversarial behavior simulation, and secure aggregation.

## Features

- **Federated Learning Core**

  - Server-client architecture
  - Multiple aggregation strategies (FedAvg, FedProx)
  - Non-IID data partitioning using Dirichlet distribution
  - Custom communication protocol

- **Client Features**

  - Heterogeneous client simulation
  - Adversarial behavior simulation
  - Local training with configurable parameters
  - Client dropout handling

- **Security & Privacy**

  - Differential privacy
  - Simulated homomorphic encryption
  - Secure aggregation
  - Byzantine fault tolerance

- **Monitoring & Visualization**
  - Real-time training dashboard
  - Client participation tracking
  - Performance metrics visualization
  - Adversarial impact analysis

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Samrawitgebremaryam/federated-learning.git
cd federated-learning
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run the basic federated learning simulation:

```bash
python src/simulation.py
```

### Advanced Configuration

The simulation supports various command-line arguments:

```bash
python src/simulation.py \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --adversarial_ratio 0.2 \
    --secure_agg True \
    --noise_scale 0.1 \
    --byzantine_threshold 0.3
```

### Monitoring Training

1. Start the training:

```bash
python src/simulation.py
```

2. Open your browser and navigate to:

```
http://localhost:5000
```

The dashboard will show:

- Global model performance
- Client participation
- Adversarial impact
- Training progress

## Project Structure

```
federated-learning/
├── src/
│   ├── client/
│   │   ├── client.py
│   │   └── adversarial_client.py
│   ├── server/
│   │   └── server.py
│   ├── data/
│   │   └── data_partition.py
│   ├── utils/
│   │   ├── communication.py
│   │   ├── secure_aggregation.py
│   │   └── visualization.py
│   └── simulation.py
├── logs/
├── results/
├── requirements.txt
└── README.md
```

##  Metrics

The system tracks and logs the following metrics:

1. **Federated Training**

   - Global model accuracy and loss
   - Round completion time
   - Client participation rate

2. **Client Heterogeneity**

   - Data distribution statistics
   - Training time variations
   - Resource utilization

3. **System Robustness**

   - Client dropout rate
   - Recovery success rate
   - Training continuation success

4. **Adversarial Behavior**

   - Attack success rate
   - Impact on model performance
   - Detection accuracy

5. **Privacy Guarantees**
   - Privacy budget consumption
   - Information leakage metrics
   - Security guarantees

## Logging and Results

- Training logs: `logs/training_*.log`
- Performance metrics: `results/training_metrics.json`
- Visualization data: `results/*.json`
- Plots: `results/plots/`

