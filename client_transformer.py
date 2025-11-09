# -*- coding: utf-8 -*-
"""
client_transformer.py

This is the client script for the BASELINE Vision Transformer (ViT) model.
It loads raw images using the *same* data_utils_cnn.py as the CNN baseline.
"""
import argparse
import warnings
import torch
import flwr as fl
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np

# Import from the CNN data utils and the NEW Transformer models
from data_utils_cnn import get_data_loaders 
from models_transformer import BaselineTransformer, train, test

# --- Constants ---
INITIAL_LR = 0.0001 # Match the stable learning rate

# --- Flower Client for Transformer Model ---
class TransformerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, optimizer, local_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        """Gets model parameters as a list of NumPy ndarrays."""
        return [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Sets model parameters from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        try:
             self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
             print(f"Warning: Error loading state dict (strict=True): {e}. Attempting non-strict.")
             self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        """Train the baseline Transformer model locally."""
        self.set_parameters(parameters)
        learning_rate = config.get("learning_rate", INITIAL_LR)
        proximal_mu = config.get("proximal_mu", 0.0)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        print(f"\n--- Client Training (Transformer Baseline, Epochs: {self.local_epochs}, LR: {learning_rate:.6f}, Mu: {proximal_mu:.2f}) ---")
        avg_train_loss, avg_train_acc = train(
            self.model, self.train_loader, epochs=self.local_epochs,
            device=self.device, proximal_mu=proximal_mu, optimizer=self.optimizer
        )
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc
        }

    def evaluate(self, parameters, config):
        """Evaluate the global Transformer model on the local validation set."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.val_loader, self.device)
        print(f"Client-side evaluation (Transformer): Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        return loss, len(self.val_loader.dataset), {
            "val_loss": loss,
            "val_accuracy": accuracy
        }

# --- Main Client Execution Block ---
def main():
    """Starts a single Transformer baseline client."""
    warnings.filterwarnings("ignore", category=UserWarning)
    
    parser = argparse.ArgumentParser(description="FeGAN Baseline Transformer Client")
    parser.add_argument("--farm-name", type=str, required=True, help="e.g., Farm_1")
    parser.add_argument("--epochs", type=int, default=10, help="Number of local training epochs per round")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Transformer client for '{args.farm_name}' on device: {device} with {args.epochs} local epochs.")

    root_path = Path("C:/Users/kusha/FeGAN_Project/data/main_data")
    
    class_map_path = root_path / "class_map.json"
    if not class_map_path.exists():
        print("Error: Universal class map 'class_map.json' not found.")
        print("Please run 'preprocess.py' on at least one farm to generate the map.")
        return
    with open(class_map_path, 'r') as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    # --- Use the CNN Data Loader ---
    train_loader, val_loader, _ = get_data_loaders(root_path, args.farm_name, class_to_idx)

    # --- Use the Baseline Transformer Model ---
    model = BaselineTransformer(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)

    client = TransformerClient(model, train_loader, val_loader, device, optimizer, args.epochs)
    
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
    
    print(f"\n--- Transformer Client {args.farm_name} finished. ---")

if __name__ == "__main__":
    main()