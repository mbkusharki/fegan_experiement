# -*- coding: utf-8 -*-
"""
data_utils.py

This is the new, definitive data utility script. Its job is to efficiently
load the pre-processed .pt graph files for a given farm, split them into
local train/val sets using stratification, and also provide a way for the
server to load a global test set using stratification.
"""
import torch
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader as GraphLoader
from pathlib import Path
import json
from sklearn.model_selection import train_test_split # For stratified split
import numpy as np # Needed for stratification
import warnings # To suppress warnings

def get_data_loaders(root_path, farm_name, class_to_idx, test_split_ratio=0.2, val_split_ratio=0.2):
    """
    Loads all pre-processed .pt graph files from a farm's 'processed'
    directory and splits them into training and validation loaders for the client.
    Uses stratified splitting to maintain class distribution.
    """
    warnings.filterwarnings("ignore", category=UserWarning) # Suppress common torch/sklearn warnings
    root_path = Path(root_path)
    processed_dir = root_path / farm_name / "processed"

    print(f"--- Loading pre-processed data for {farm_name} from: {processed_dir} ---")

    graph_files = list(processed_dir.glob("*.pt"))
    if not graph_files:
        raise FileNotFoundError(f"No .pt files found in {processed_dir}. Please run preprocess.py.")

    all_graphs = []
    all_labels = []
    print(f"Loading {len(graph_files)} graph files...")
    for f in graph_files:
        try:
            # Add weights_only=False for security in newer PyTorch versions
            graph = torch.load(f, weights_only=False)
            all_graphs.append(graph)
            all_labels.append(graph.y.item()) # Get the label for stratification
        except Exception as e:
            print(f"Warning: Could not load graph file {f}. Reason: {e}. Skipping.")


    if not all_graphs:
        raise ValueError(f"Loaded 0 valid graphs from {processed_dir}. Please check the files or run preprocess.py again.")

    print(f"Loaded {len(all_graphs)} valid graphs.")
    num_classes = len(class_to_idx)

    if len(all_graphs) < 2: # Need at least 2 samples for splitting
        print("Warning: Not enough data to create train/val splits. Using all data for training.")
        train_loader = GraphLoader(all_graphs, batch_size=32, shuffle=True)
        val_loader = GraphLoader([], batch_size=32, shuffle=False)
        return train_loader, val_loader, num_classes

    indices = list(range(len(all_graphs)))
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    min_samples = counts.min() if len(counts) > 0 else 0

    # Calculate number of splits required for train/val from the client's portion
    client_val_ratio_adjusted = val_split_ratio / (1 - test_split_ratio)

    if min_samples >= 2 and len(unique_labels) > 1 : # Need at least 2 samples AND > 1 class
        try:
            # Split off global test set first (indices not used by client)
            train_val_indices, _ = train_test_split(
                indices,
                test_size=test_split_ratio,
                stratify=all_labels,
                random_state=42
            )

            # Split client's portion into train and val
            client_labels = [all_labels[i] for i in train_val_indices]
            unique_client_labels, client_counts = np.unique(client_labels, return_counts=True)
            min_client_samples = client_counts.min() if len(client_counts) > 0 else 0

            if min_client_samples >= 2 and len(unique_client_labels) > 1:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=client_val_ratio_adjusted,
                    stratify=client_labels,
                    random_state=42
                )
            else: # Fallback if client data lacks diversity for stratification
                 print("Warning: Client data has < 2 samples per class or only 1 class. Using random split for train/val.")
                 num_val_client = int(len(train_val_indices) * client_val_ratio_adjusted)
                 val_indices = train_val_indices[:num_val_client]
                 train_indices = train_val_indices[num_val_client:]


        except ValueError as e: # Catch stratification errors
            print(f"Warning: Stratification failed ({e}), falling back to random split for client data.")
            num_client_data = int(len(all_graphs) * (1 - test_split_ratio))
            num_val_client = int(num_client_data * client_val_ratio_adjusted)
            val_indices = indices[:num_val_client] # Arbitrary split
            train_indices = indices[num_val_client:num_client_data]

    else: # Fallback for very small overall datasets or only one class
        print("Warning: Dataset has < 2 samples per class or only 1 class overall. Using random split.")
        num_client_data = int(len(all_graphs) * (1 - test_split_ratio))
        num_val_client = int(num_client_data * client_val_ratio_adjusted)
        val_indices = indices[:num_val_client]
        train_indices = indices[num_val_client:num_client_data]

    # --- Important: Use Subsets for DataLoaders ---
    # Create subsets from the original list based on indices
    train_dataset = Subset(all_graphs, train_indices)
    val_dataset = Subset(all_graphs, val_indices)

    # Pass the Subset objects directly to the GraphLoader
    train_loader = GraphLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = GraphLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Created train loader with {len(train_dataset)} graphs and val loader with {len(val_dataset)} graphs.")

    return train_loader, val_loader, num_classes


def get_global_test_loader(root_path, class_to_idx, test_split_ratio=0.2):
    """
    Loads a portion of pre-processed .pt graph files from ALL farms
    to create a global test set. Uses stratification.
    """
    warnings.filterwarnings("ignore", category=UserWarning) # Suppress common torch/sklearn warnings
    root_path = Path(root_path)
    all_graphs = []
    all_labels = []

    print("--- Loading data from ALL farms for GLOBAL test set ---")
    farm_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("Farm_")]
    for farm_dir in farm_dirs:
        processed_dir = farm_dir / "processed"
        graph_files = list(processed_dir.glob("*.pt"))
        print(f"Loading {len(graph_files)} graphs from {farm_dir.name}...")
        for f in graph_files:
             try:
                # Add weights_only=False for security in newer PyTorch versions
                graph = torch.load(f, weights_only=False)
                all_graphs.append(graph)
                all_labels.append(graph.y.item())
             except Exception as e:
                 print(f"Warning: Could not load graph file {f} for global test set. Reason: {e}. Skipping.")


    if not all_graphs:
        raise FileNotFoundError("No processed graph files found in any farm directory.")

    print(f"Loaded a total of {len(all_graphs)} valid graphs from all farms.")

    indices = list(range(len(all_graphs)))
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    min_samples = counts.min() if len(counts) > 0 else 0

    if min_samples >= 2 and len(unique_labels) > 1 :
         try:
             _, test_indices = train_test_split(
                 indices,
                 test_size=test_split_ratio,
                 stratify=all_labels,
                 random_state=42 # Use the same random state as client split
             )
         except ValueError as e:
             print(f"Warning: Stratification failed for global test set ({e}), falling back to random split.")
             num_test = int(len(all_graphs) * test_split_ratio)
             test_indices = indices[-num_test:] # Take the end portion
    else:
         print("Warning: Not enough samples per class or only 1 class for stratification, using random split for test set.")
         num_test = int(len(all_graphs) * test_split_ratio)
         test_indices = indices[-num_test:]

    # Create Subset for test data
    test_dataset = Subset(all_graphs, test_indices)
    test_loader = GraphLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Created global test set with {len(test_dataset)} graphs.")

    return test_loader, len(class_to_idx)

