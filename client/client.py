import os
from flwr.client import start_client, NumPyClient
from flwr.common import Context
from utils import load_trainset, load_testset, set_weights, get_weights, SimpleModel, train_model, evaluate_model

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# -----------------------------------------------------------------------------
# Flower client implementation
# -----------------------------------------------------------------------------

class FlowerClient(NumPyClient):
    def __init__(self, model, trainset, testset):
        self.model = model
        self.trainset = trainset
        self.testset = testset

    def get_parameters(self, config):
        # Return model parameters as list of NumPy arrays
        return get_weights(self.model)

    def fit(self, parameters, config):
        # Update local model
        set_weights(self.model, parameters)
        # Train locally
        train_model(self.model, self.trainset)
        # Return updated weights and training size
        return get_weights(self.model), len(self.trainset), {}

    def evaluate(self, parameters, config):
        # Update local model
        set_weights(self.model, parameters)
        # Evaluate locally
        loss, accuracy = evaluate_model(self.model, self.testset)
        # Return loss, test size, and metrics
        return loss, len(self.testset), {"accuracy": accuracy}


def client_fn(cid: str) -> NumPyClient:
    # Load local data partition and shared test set
    trainset = load_trainset()
    testset = load_testset()

    # Initialize model
    net = SimpleModel()
    return FlowerClient(net, trainset, testset)


if __name__ == "__main__":
    # Address of the Flower server
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    # Start Flower client
    start_client(server_address=server_address, client=client_fn)
