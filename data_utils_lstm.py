# -*- coding: utf-8 -*-
"""
data_utils_lstm.py

This is the data loader for the BASELINE LSTM model.
It loads the *pre-processed .pt graph files* but extracts their
node features (x) and labels (y), padding them into sequences
for the LSTM.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

# --- Custom Dataset for Loading Graph Features as Sequences ---
class PlantGraphSequenceDataset(Dataset):
    """
    A custom dataset to load pre-processed .pt graph files
    and return their node features (x) and label (y).
    """
    def __init__(self, graph_files):
        self.graph_files = graph_files

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        try:
            graph = torch.load(self.graph_files[idx], weights_only=False)
            # x shape: [num_nodes, 1280], y shape: [1]
            return graph.x, graph.y.squeeze() 
        except Exception as e:
            print(f"Warning: Could not load graph file {self.graph_files[idx]}. Reason: {e}. Skipping.")
            return None, None

def robust_collate_fn_lstm(batch):
    """
    A collate_fn that filters out None (corrupted) samples
    and pads sequences to the same length.
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None

    features, labels = zip(*batch)
    
    # Pad sequences (features) to the length of the longest seq in the batch
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Stack labels
    labels = torch.stack(labels)
    
    return features_padded, labels

# --- Client-side Data Loader ---
def get_data_loaders(root_path, farm_name, class_to_idx, test_split_ratio=0.2, val_split_ratio=0.2):
    """
    Loads all pre-processed .pt graph files from a farm's 'processed'
    directory and splits them into training and validation loaders for the LSTM.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    root_path = Path(root_path)
    processed_dir = root_path / farm_name / "processed"
    
    print(f"--- Loading pre-processed data for {farm_name} (LSTM) from: {processed_dir} ---")
    
    all_graph_files = list(processed_dir.glob("*.pt"))
    
    if not all_graph_files:
        raise FileNotFoundError(f"No .pt files found in {processed_dir}. Please run preprocess.py.")

    print(f"Found {len(all_graph_files)} graph files.")
    
    # Get labels for stratification
    all_labels = []
    valid_files = []
    for f in all_graph_files:
        try:
            # Load just to get the label
            graph = torch.load(f, weights_only=False)
            all_labels.append(graph.y.item())
            valid_files.append(f)
        except Exception:
            continue # Skip corrupted files
            
    all_graph_files = valid_files
    num_classes = len(class_to_idx)

    if len(all_graph_files) < 2:
        print("Warning: Not enough data for splits. Using all data for training.")
        train_dataset = PlantGraphSequenceDataset(all_graph_files)
        val_dataset = PlantGraphSequenceDataset([])
    else:
        indices = list(range(len(all_graph_files)))
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        min_samples = counts.min() if len(counts) > 0 else 0
        client_val_ratio_adjusted = val_split_ratio / (1 - test_split_ratio)

        if min_samples >= 2 and len(unique_labels) > 1:
            try:
                train_val_indices, _ = train_test_split(
                    indices, test_size=test_split_ratio, stratify=all_labels, random_state=42
                )
                client_labels = [all_labels[i] for i in train_val_indices]
                unique_client_labels, client_counts = np.unique(client_labels, return_counts=True)
                min_client_samples = client_counts.min() if len(client_counts) > 0 else 0

                if min_client_samples >= 2 and len(unique_client_labels) > 1:
                    train_indices, val_indices = train_test_split(
                        train_val_indices, test_size=client_val_ratio_adjusted, stratify=client_labels, random_state=42
                    )
                else:
                    print("Warning: Client data has < 2 samples per class. Using random split for train/val.")
                    num_val_client = int(len(train_val_indices) * client_val_ratio_adjusted)
                    val_indices = train_val_indices[:num_val_client]
                    train_indices = train_val_indices[num_val_client:]
            except ValueError as e:
                print(f"Warning: Stratification failed ({e}), falling back to random split.")
                num_client_data = int(len(all_graph_files) * (1-test_split_ratio))
                num_val_client = int(num_client_data * client_val_ratio_adjusted)
                val_indices = indices[:num_val_client]
                train_indices = indices[num_val_client:num_client_data]
        else:
            print("Warning: Dataset has < 2 samples per class. Using random split.")
            num_client_data = int(len(all_graph_files) * (1-test_split_ratio))
            num_val_client = int(num_client_data * client_val_ratio_adjusted)
            val_indices = indices[:num_val_client]
            train_indices = indices[num_val_client:num_client_data]

        train_paths = [all_graph_files[i] for i in train_indices]
        val_paths = [all_graph_files[i] for i in val_indices]
        
        train_dataset = PlantGraphSequenceDataset(train_paths)
        val_dataset = PlantGraphSequenceDataset(val_paths)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=robust_collate_fn_lstm)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=robust_collate_fn_lstm)
    
    print(f"Created train loader with {len(train_dataset)} sequences and val loader with {len(val_dataset)} sequences.")
    return train_loader, val_loader, num_classes