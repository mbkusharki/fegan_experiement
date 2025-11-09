# -*- coding: utf-8 -*-
"""
client.py

This is the definitive, high-performance version of the FeGAN client.
This final version includes the fix for the .numpy() RuntimeError by
using .detach() before converting tensors.
"""
import argparse
import warnings
import torch
import flwr as fl
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np

from data_utils import get_data_loaders
from models import FeGAN, train, test

# --- Constants ---
INITIAL_LR = 0.0001

# --- Flower Client with Sparsification ---
class FeGANClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, optimizer, local_epochs, sparsity_level):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer
        self.local_epochs = local_epochs
        self.sparsity_level = sparsity_level # Percentage of updates to ZERO OUT (e.g., 0.9 means keep 10%)

    def get_parameters(self, config):
        """Gets model parameters as a list of NumPy ndarrays."""
        # Ensure parameters are detached before converting to numpy
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
        """Train locally, sparsify the update, reconstruct, and return parameters."""
        # --- THIS IS THE FIX: Added .detach() ---
        initial_parameters_list = [p.clone().detach().cpu().numpy() for p in self.model.parameters()]
        
        self.set_parameters(parameters) # Load global parameters

        learning_rate = config.get("learning_rate", INITIAL_LR)
        proximal_mu = config.get("proximal_mu", 0.0)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        print(f"\n--- Client Training (Epochs: {self.local_epochs}, LR: {learning_rate:.6f}, Mu: {proximal_mu:.2f}) ---")
        avg_train_loss, avg_train_acc = train(
            self.model, self.train_loader, epochs=self.local_epochs,
            device=self.device, proximal_mu=proximal_mu, optimizer=self.optimizer
        )

        # --- Sparsification Logic ---
        # Ensure final parameters are also detached
        final_parameters_list = [p.clone().detach().cpu().numpy() for p in self.model.parameters()]
        
        # 1. Calculate the update (difference)
        # Ensure lists have the same length (they should if model structure is consistent)
        if len(final_parameters_list) != len(initial_parameters_list):
             print("Warning: Parameter list length mismatch before/after training. Skipping sparsification.")
             # Fallback: return the final parameters directly if mismatch occurs
             # This should ideally not happen if set_parameters works correctly.
             return self.get_parameters(config={}), len(self.train_loader.dataset), {
                 "train_loss": avg_train_loss,
                 "train_accuracy": avg_train_acc
             }

        updates = [final - initial for final, initial in zip(final_parameters_list, initial_parameters_list)]
        
        # 2. Flatten all updates to find the global threshold
        all_updates_flat = np.concatenate([upd.flatten() for upd in updates])
        threshold = np.percentile(np.abs(all_updates_flat), self.sparsity_level * 100)
        
        # 3. Sparsify: zero out updates below the threshold
        sparse_updates = []
        total_params = 0
        non_zero_params = 0
        for upd in updates:
            sparse_upd = upd * (np.abs(upd) >= threshold)
            sparse_updates.append(sparse_upd)
            total_params += upd.size
            non_zero_params += np.count_nonzero(sparse_upd)
            
        sparsity_achieved = 1.0 - (non_zero_params / total_params) if total_params > 0 else 0
        print(f"Sparsifying update: Target={self.sparsity_level*100:.1f}%, Achieved={sparsity_achieved*100:.1f}% ({non_zero_params}/{total_params} non-zero)")

        # 4. Reconstruct parameters to send back (initial *global* + sparse_update)
        # Ensure parameters list (from server) and sparse_updates have same length
        if len(parameters) != len(sparse_updates):
             print("Warning: Global parameters and sparse update length mismatch. Skipping reconstruction.")
             # Fallback: return the final (unsparsified but trained) parameters
             return self.get_parameters(config={}), len(self.train_loader.dataset), {
                 "train_loss": avg_train_loss,
                 "train_accuracy": avg_train_acc
             }
             
        reconstructed_params_list = [p_glob + sparse_upd for p_glob, sparse_upd in zip(parameters, sparse_updates)]
        
        # NOTE: In a real system with SecAgg, only the sparse_updates (or an encoded
        # version) would be sent, protected cryptographically. Here, we send the
        # reconstructed dense parameters to fit the standard Flower API.

        return reconstructed_params_list, len(self.train_loader.dataset), {
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc
        }

    def evaluate(self, parameters, config):
        """Evaluate the global model on this client's local validation set."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.val_loader, self.device)
        print(f"Client-side evaluation: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        return loss, len(self.val_loader.dataset), {
            "val_loss": loss,
            "val_accuracy": accuracy
        }

# --- Main Client Execution Block ---
def main():
    """Starts a single, self-contained FeGAN client with sparsification."""
    warnings.filterwarnings("ignore", category=UserWarning)
    
    parser = argparse.ArgumentParser(description="FeGAN Client (Sparse Updates)")
    parser.add_argument("--farm-name", type=str, required=True, help="e.g., Farm_1")
    parser.add_argument("--epochs", type=int, default=10, help="Local training epochs")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Sparsity level (0.0 to 1.0), e.g., 0.9 means keep 10%")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting client for '{args.farm_name}' on device: {device} with {args.epochs} local epochs, Sparsity: {args.sparsity*100:.1f}%")

    root_path = Path("C:/Users/kusha/FeGAN_Project/data/main_data")
    
    processed_dir = root_path / args.farm_name / "processed"
    if not processed_dir.exists() or not any(processed_dir.iterdir()):
        print(f"\nError: Pre-processed data not found for {args.farm_name}.")
        print(f"Run preprocess.py first: python preprocess.py --farm-name \"{args.farm_name}\"\n")
        return

    class_map_path = root_path / "class_map.json"
    with open(class_map_path, 'r') as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    train_loader, val_loader, _ = get_data_loaders(root_path, args.farm_name, class_to_idx)

    model = FeGAN(in_channels=1280, hidden_channels=512, out_channels=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR)

    client = FeGANClient(model, train_loader, val_loader, device, optimizer, args.epochs, args.sparsity)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
    
    print(f"\n--- Client {args.farm_name} finished. ---")

if __name__ == "__main__":
    main()
