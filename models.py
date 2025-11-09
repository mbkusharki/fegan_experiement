# -*- coding: utf-8 -*-
"""
models.py

This is the definitive, high-capacity version of the FeGAN model.
This final version increases dropout rates significantly to combat
overfitting and improve generalization, which is the key to breaking
through the final accuracy plateau.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from collections import OrderedDict
from tqdm import tqdm
import numpy as np # Needed for confusion matrix data

# --- Definitive High-Capacity FeGAN Model (Stronger Dropout) ---
class FeGAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout_rate=0.7): # Increased default dropout
        super(FeGAN, self).__init__()
        self.dropout_rate = dropout_rate
        # Input channels must match the feature extractor's output (1280)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_rate)
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout_rate)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply dropout *before* the first layer as well
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.gat2(x, edge_index))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat3(x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        # Apply dropout before the final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
        
    def set_weights(self, parameters):
        """Helper to set model weights from a list of NumPy arrays."""
        state_dict = OrderedDict()
        param_keys = list(self.state_dict().keys())
        for i, key in enumerate(param_keys):
            if i < len(parameters):
                 state_dict[key] = torch.tensor(parameters[i])
            else:
                 print(f"Warning: Missing parameter in loaded state for key {key}")
                 state_dict[key] = self.state_dict()[key]
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            print("Attempting non-strict loading...")
            self.load_state_dict(state_dict, strict=False)


# --- Test Function (for Client-side Validation) ---
# (No changes needed in test function)
def test(model, graph_loader, device):
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for graph_batch in graph_loader:
            if graph_batch is None: continue
            graph_batch = graph_batch.to(device)
            outputs = model(graph_batch)
            batch_loss = criterion(outputs, graph_batch.y)
            total_loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += graph_batch.y.size(0)
            correct += (predicted == graph_batch.y).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(graph_loader) if len(graph_loader) > 0 else 0
    return avg_loss, accuracy

# --- Train Function (Returns Metrics) ---
# (No changes needed in train function, gradient clipping already present)
def train(model, graph_loader, epochs, device, optimizer, proximal_mu=0.0):
    criterion = nn.CrossEntropyLoss()
    global_model_weights = [param.clone().detach() for param in model.parameters() if param.requires_grad]
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        pbar = tqdm(graph_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for graph_batch in pbar:
            if graph_batch is None: continue
            graph_batch = graph_batch.to(device)
            optimizer.zero_grad()
            outputs = model(graph_batch)
            loss = criterion(outputs, graph_batch.y)

            if proximal_mu > 0:
                proximal_term = 0.0
                for local_weight, global_weight in zip(filter(lambda p: p.requires_grad, model.parameters()), global_model_weights):
                    proximal_term += (local_weight - global_weight).pow(2).sum()
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_samples = graph_batch.y.size(0)
            epoch_samples += batch_samples
            epoch_correct += (predicted == graph_batch.y).sum().item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(graph_loader) if len(graph_loader) > 0 else 0
        epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    avg_train_loss = total_loss / (len(graph_loader) * epochs) if len(graph_loader) > 0 else 0
    avg_train_acc = total_correct / total_samples if total_samples > 0 else 0
    return avg_train_loss, avg_train_acc


# --- Test Function for Confusion Matrix (Returns Predictions) ---
# (No changes needed in test_for_cm function)
def test_for_cm(model, graph_loader, device):
    """Runs evaluation and returns true labels and predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for graph_batch in tqdm(graph_loader, desc="Generating predictions for CM"):
            if graph_batch is None: continue
            graph_batch = graph_batch.to(device)
            outputs = model(graph_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(graph_batch.y.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

