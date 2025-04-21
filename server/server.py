import os
import torch
from flwr.server import start_server
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from utils import SimpleModel, get_weights, evaluate_model, compute_confusion_matrix, plot_confusion_matrix, load_testset, set_weights
from logging import INFO
from flwr.common.logger import log

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Load and initialize the model
net = SimpleModel()
initial_parameters = ndarrays_to_parameters(get_weights(net))

testset = load_testset()

def evaluate(server_round, parameters, config):
    set_weights(net, parameters)

    _, accuracy = evaluate_model(net, testset)

    log(INFO, f"[Round {server_round}] Test accuracy on all digits: {accuracy:.4f}")

    if server_round == config["num_rounds"]:
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model")

    return 0.0, {"accuracy": accuracy}  # Dummy loss

# Configure federated strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.0,
    initial_parameters=initial_parameters,
    evaluate_fn=evaluate,
)

# Server config
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", 3))

if __name__ == "__main__":
    start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": NUM_ROUNDS},
        strategy=strategy,
    )
