# -*- coding: utf-8 -*-
"""
models_lstm.py

This module defines the BASELINE LSTM model.
It is designed to take the *same 1280-d features* as FeGAN,
but treats them as a simple sequence, ignoring the graph structure.
This provides a direct test of the GAT's architectural benefit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

# --- Definitive Baseline LSTM Model ---
class BaselineLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BaselineLSTM, self).__init__()
        # in_channels = 1280 (feature dim)
        # hidden_channels = 512 (to match FeGAN's capacity)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=2, # A 2-layer LSTM is a strong baseline
            batch_first=True, # Input shape is [batch_size, seq_len, features]
            dropout=0.5,
            bidirectional=True # Bidirectional for more power
        )
        
        # Classifier takes the final hidden state
        # (hidden_channels * 2 because it's bidirectional)
        self.classifier = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x):
        # x shape: [batch_size, seq_len, 1280]
        # output shape: [batch_size, seq_len, hidden_size * 2]
        # hn, cn shapes: [num_layers * 2, batch_size, hidden_size]
        output, (hn, cn) = self.lstm(x)
        
        # We take the final hidden state from the last layer
        # Concatenate the final forward (hn[-2]) and backward (hn[-1]) hidden states
        final_hidden_state = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        # Pass through the classifier
        x = F.dropout(final_hidden_state, p=0.6, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
        
    def set_weights(self, parameters):
        """Helper to set model weights from a list of NumPy arrays."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            print("Attempting non-strict loading...")
            self.load_state_dict(state_dict, strict=False)

# --- Test Function (for Client-side Sequence Evaluation) ---
def test(model, data_loader, device):
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for sequences, labels in data_loader:
            if sequences is None: continue # Skip empty batches
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            batch_loss = criterion(outputs, labels)
            total_loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    return avg_loss, accuracy

# --- Train Function (for Client-side Sequence Training) ---
def train(model, data_loader, epochs, device, optimizer, proximal_mu=0.0):
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
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for sequences, labels in pbar:
            if sequences is None: continue
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

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
            batch_samples = labels.size(0)
            epoch_samples += batch_samples
            epoch_correct += (predicted == labels).sum().item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(data_loader) if len(data_loader) > 0 else 0
        epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    avg_train_loss = total_loss / (len(data_loader) * epochs) if len(data_loader) > 0 else 0
    avg_train_acc = total_correct / total_samples if total_samples > 0 else 0
    return avg_train_loss, avg_train_acc

# --- Test Function for Confusion Matrix (Returns Predictions) ---
def test_for_cm(model, data_loader, device):
    """Runs evaluation and returns true labels and predictions for sequences."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in tqdm(data_loader, desc="Generating predictions for CM"):
            if sequences is None: continue
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)