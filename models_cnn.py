# -*- coding: utf-8 -*-
"""
models_cnn.py

This module defines the BASELINE CNN model and its training/testing loops.
To ensure a fair comparison with FeGAN, this model uses the same
EfficientNet-B0 backbone, but with a simple linear classifier instead of a GAT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

# --- Definitive Baseline CNN Model ---
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super(BaselineCNN, self).__init__()
        # Load pre-trained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.feature_extractor = models.efficientnet_b0(weights=weights)
        
        # Get the number of input features for the classifier
        in_features = self.feature_extractor.classifier[1].in_features
        
        # Replace the classifier with a new one for our number of classes
        self.feature_extractor.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.feature_extractor(x)
        
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

# --- Test Function (for Client-side Image Evaluation) ---
def test(model, data_loader, device):
    """Evaluates the CNN model on a standard image data loader."""
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            if images.nelement() == 0: continue # Skip empty batches
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            total_loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    return avg_loss, accuracy

# --- Train Function (for Client-side Image Training) ---
def train(model, data_loader, epochs, device, optimizer, proximal_mu=0.0):
    """Trains the CNN model with FedProx and Gradient Clipping."""
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
        for images, labels in pbar:
            if images.nelement() == 0: continue
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Add FedProx term
            if proximal_mu > 0:
                proximal_term = 0.0
                for local_weight, global_weight in zip(filter(lambda p: p.requires_grad, model.parameters()), global_model_weights):
                    proximal_term += (local_weight - global_weight).pow(2).sum()
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
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
    """Runs evaluation and returns true labels and predictions for images."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Generating predictions for CM"):
            if images.nelement() == 0: continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)