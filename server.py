# -*- coding: utf-8 -*-
"""
server.py

This is the definitive, experiment-ready "blind" FeGAN server.
This version implements CONTINUAL LEARNING: it loads the previously saved
best model parameters if they exist, allowing training to resume and improve
over multiple sessions.
"""
import flwr as fl
import torch
from models import FeGAN, test_for_cm # Import the new test function
from data_utils import get_global_test_loader # Function to load the test set
import json
from pathlib import Path
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Metrics
from typing import List, Tuple, Dict
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings # Import warnings
from collections import OrderedDict # Import OrderedDict

# --- Constants ---
NUM_ROUNDS = 30 # Reverted as requested
INITIAL_LR = 0.0001

# --- Metrics Aggregation Functions ---
def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics (like accuracy, loss) using weighted average."""
    num_total_examples = sum([num_examples for num_examples, _ in metrics])
    if num_total_examples == 0: return {}
    aggregated_metrics = {}
    # Aggregate each metric type (e.g., accuracy, val_loss)
    # Check if the first metric tuple has metrics before accessing keys
    if metrics and metrics[0][1]:
         metric_keys = metrics[0][1].keys() # Get keys from the first client's metrics
         for key in metric_keys:
              # Ensure metric value is float before multiplication
              weighted_sum = sum([num_examples * float(m.get(key, 0.0)) for num_examples, m in metrics])
              aggregated_metrics[key] = weighted_sum / num_total_examples
    return aggregated_metrics

# --- Custom Adaptive Strategy ---
class AdaptiveFedProx(fl.server.strategy.FedProx):
    def __init__(self, proximal_mu_arg, initial_parameters, *args, **kwargs): # Pass initial_parameters here
        # Note: We pass initial_parameters to the parent class now
        super().__init__(*args, proximal_mu=proximal_mu_arg, initial_parameters=initial_parameters, **kwargs)
        self.best_accuracy = 0.0
        self.learning_rate = INITIAL_LR
        self.patience_counter = 0
        self.patience_limit = 5
        self.best_parameters = initial_parameters # Start with loaded/initial params
        # Store history
        self.history: Dict[str, List[Tuple[int, float]]] = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
        # Keep track of previous accuracy for saving logic
        self._previous_best_accuracy = 0.0


    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, ins in client_instructions:
            ins.config["learning_rate"] = self.learning_rate
        return client_instructions

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Log aggregated training metrics from clients
        if aggregated_metrics:
             train_loss = aggregated_metrics.get("train_loss")
             train_acc = aggregated_metrics.get("train_accuracy")
             if train_loss is not None: self.history["train_loss"].append((server_round, train_loss))
             if train_acc is not None: self.history["train_accuracy"].append((server_round, train_acc))
             print(f"Round {server_round} - Aggregated Train Loss: {train_loss if train_loss is not None else 'N/A':.4f}, "
                   f"Train Accuracy: {train_acc if train_acc is not None else 'N/A':.4f}")

        if aggregated_parameters is not None:
             # Store the latest parameters; saving happens after evaluation
             self.current_round_parameters = aggregated_parameters

        return aggregated_parameters, aggregated_metrics


    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if aggregated_metrics and "val_accuracy" in aggregated_metrics:
            accuracy = aggregated_metrics["val_accuracy"]
            loss = aggregated_metrics.get("val_loss", None)

            if loss is not None: self.history["val_loss"].append((server_round, loss))
            if accuracy is not None: self.history["val_accuracy"].append((server_round, accuracy))
            print(f"Round {server_round} - Aggregated Val Loss: {loss if loss is not None else 'N/A':.4f}, "
                  f"Val Accuracy: {accuracy:.4f}")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.patience_counter = 0
                print(f"New best accuracy: {self.best_accuracy:.4f}. Saving model parameters...")
                # Save the parameters from the *previous* fit round
                if hasattr(self, 'current_round_parameters') and self.current_round_parameters is not None:
                     np_params = parameters_to_ndarrays(self.current_round_parameters)
                     torch.save(np_params, f"best_fegan_model_mu_{self.proximal_mu}.pth")
                     self.best_parameters = self.current_round_parameters # Update the best known parameters
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience_limit:
                self.learning_rate *= 0.1
                self.patience_counter = 0
                print(f"\n--- Accuracy plateaued. Reducing learning rate to {self.learning_rate:.6f} ---\n")

        return aggregated_loss, aggregated_metrics

# --- Confusion Matrix Plotting ---
def plot_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    plt.close()

# --- Main Server Execution Block ---
def main():
    """Starts the definitive 'blind' FeGAN server with continual learning."""
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="FeGAN Server (Tunable Mu, Full Eval, Continual Learning)")
    parser.add_argument("--mu", type=float, default=0.1, help="Proximal term constant (mu) for FedProx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting FeGAN Server (Mu={args.mu}, Device={device}) ---")

    root_path = Path("C:/Users/kusha/FeGAN_Project/data/main_data")

    class_map_path = root_path / "class_map.json"
    if not class_map_path.exists():
        print("Error: Universal class map 'class_map.json' not found.")
        print("Please run 'preprocess.py' once on any farm to generate it.")
        return
    with open(class_map_path, 'r') as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # --- Continual Learning Logic ---
    best_model_path = f"best_fegan_model_mu_{args.mu}.pth"
    initial_parameters = None
    if Path(best_model_path).exists():
        print(f"\n--- Found existing model parameters ({best_model_path}). Resuming training. ---\n")
        try:
            # Load the NumPy arrays
            loaded_params_nd = torch.load(best_model_path, weights_only=False)
            # Convert them to Flower Parameters
            initial_parameters = ndarrays_to_parameters(loaded_params_nd)
        except Exception as e:
            print(f"Warning: Could not load existing parameters from {best_model_path}. Starting from scratch. Error: {e}")
            initial_parameters = None # Fallback to starting fresh

    if initial_parameters is None:
        print("\n--- No existing model found or load failed. Initializing new model parameters. ---\n")
        # Initialize the high-capacity model if no saved parameters are loaded
        initial_model = FeGAN(in_channels=1280, hidden_channels=512, out_channels=num_classes)
        initial_parameters_nd = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
        initial_parameters = ndarrays_to_parameters(initial_parameters_nd)

    # --- Strategy Definition ---
    strategy = AdaptiveFedProx(
        proximal_mu_arg=args.mu,
        initial_parameters=initial_parameters, # Pass loaded or new initial parameters
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )

    print("\nStarting Flexible FeGAN Federated Server on 0.0.0.0:8080")
    print("Server will start training as soon as 1 or more clients connect...")

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\n--- Server finished training ---")
    print(f"Final best accuracy achieved across clients: {strategy.best_accuracy:.4f}")

    # --- Final Evaluation and Confusion Matrix ---
    # Use the best parameters stored in the strategy after the run
    if strategy.best_parameters is not None:
        print(f"\n--- Evaluating best model achieved during training (Accuracy: {strategy.best_accuracy:.4f}) on global test set ---")
        best_params_nd = parameters_to_ndarrays(strategy.best_parameters) # Convert from Flower Parameters

        test_loader, _ = get_global_test_loader(root_path, class_to_idx)

        final_model = FeGAN(in_channels=1280, hidden_channels=512, out_channels=num_classes).to(device)

        params_dict = zip(final_model.state_dict().keys(), best_params_nd)
        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
        final_model.load_state_dict(state_dict, strict=True)

        true_labels, predictions = test_for_cm(final_model, test_loader, device)

        cm = confusion_matrix(true_labels, predictions)
        class_names = [idx_to_class[i] for i in range(num_classes)]
        plot_confusion_matrix(cm, class_names, filename=f"confusion_matrix_mu_{args.mu}.png")
    else:
        # Check if the initially loaded model (if any) should be evaluated
        if Path(best_model_path).exists() and initial_parameters is not None and strategy.best_accuracy == 0:
             print(f"\n--- Training did not improve accuracy. Evaluating the initially loaded model ({best_model_path}) on global test set ---")
             initial_params_nd = parameters_to_ndarrays(initial_parameters) # Use the initially loaded params
             test_loader, _ = get_global_test_loader(root_path, class_to_idx)
             final_model = FeGAN(in_channels=1280, hidden_channels=512, out_channels=num_classes).to(device)
             params_dict = zip(final_model.state_dict().keys(), initial_params_nd)
             state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
             final_model.load_state_dict(state_dict, strict=True)
             true_labels, predictions = test_for_cm(final_model, test_loader, device)
             cm = confusion_matrix(true_labels, predictions)
             class_names = [idx_to_class[i] for i in range(num_classes)]
             plot_confusion_matrix(cm, class_names, filename=f"confusion_matrix_mu_{args.mu}_initial.png")

        else:
             print("\nNo best model was achieved during training and no previous model existed.")


    # Plot training history
    if strategy.history["val_accuracy"]:
         rounds, accs = zip(*[(r, a) for r, a in strategy.history["val_accuracy"] if a is not None])
         if rounds:
             plt.figure()
             plt.plot(rounds, accs, marker='o', linestyle='-')
             plt.title(f'Validation Accuracy per Round (Mu={args.mu})')
             plt.xlabel('Round')
             plt.ylabel('Accuracy')
             plt.grid(True)
             plt.ylim(0, 1) # Set y-axis limit for accuracy
             plt.xticks(range(min(rounds), max(rounds)+1)) # Ensure integer round numbers on x-axis
             plt.savefig(f"validation_accuracy_mu_{args.mu}.png")
             print(f"Validation accuracy plot saved to validation_accuracy_mu_{args.mu}.png")
             plt.close()


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import numpy as np
    except ImportError:
        print("\nPlease install matplotlib, seaborn, scikit-learn, and numpy for plotting:")
        print("pip install matplotlib seaborn scikit-learn numpy\n")
        exit()

    main()