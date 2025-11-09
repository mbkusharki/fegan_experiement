# -*- coding: utf-8 -*-
"""
data_utils_cnn.py

This is the data loader for the BASELINE CNN model.
It does NOT load graphs. Instead, it loads the ORIGINAL raw images
from the farm folders and applies standard torchvision transforms.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
import numpy as np
import warnings
from tqdm import tqdm

# --- Standard Image Transforms for CNNs ---
# Get the standard transforms for EfficientNet-B0
try:
    from torchvision.models import EfficientNet_B0_Weights
    EFFICIENTNET_TRANSFORMS = EfficientNet_B0_Weights.DEFAULT.transforms()
except ImportError: # Fallback for older torchvision
    EFFICIENTNET_TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# We add augmentations for the training set
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
VAL_TRANSFORM = EFFICIENTNET_TRANSFORMS


# --- Custom Dataset for Loading Images ---
class PlantImageDataset(Dataset):
    """
    A custom dataset to load raw image files from a list of paths
    and apply the correct transforms and unified labels.
    """
    def __init__(self, file_paths, class_to_idx, transform=None):
        self.file_paths = file_paths
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.unified_labels = self._get_labels()
        
    def _get_labels(self):
        labels = []
        for img_path in self.file_paths:
            class_name = Path(img_path).parent.name
            unified_class_name = "Healthy" if "healthy" in class_name.lower() else class_name
            labels.append(self.class_to_idx[unified_class_name])
        return labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.unified_labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: Skipping corrupted image {img_path}. Error: {e}")
            # Return a dummy tensor and label, will be filtered by collate_fn
            return None, -1

def robust_collate_fn(batch):
    """A collate_fn that filters out None (corrupted) samples."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)


# --- Client-side Data Loader ---
def get_data_loaders(root_path, farm_name, class_to_idx, test_split_ratio=0.2, val_split_ratio=0.2):
    """
    Loads all RAW images from a farm's directory, applies transforms,
    and splits them into training and validation loaders.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    root_path = Path(root_path)
    farm_path = root_path / farm_name
    
    print(f"--- Loading RAW IMAGES for {farm_name} from: {farm_path} ---")

    # Get all image files, excluding the 'processed' directory
    all_image_paths = [
        p for p in farm_path.glob('*/*.*') 
        if p.parent.name != 'processed'
    ]
    
    if not all_image_paths:
        raise FileNotFoundError(f"No raw image files found in {farm_path}. Check directory structure.")

    print(f"Found {len(all_image_paths)} raw images.")

    # Create a temporary dataset to get labels for stratification
    temp_dataset = PlantImageDataset(all_image_paths, class_to_idx, transform=None)
    all_labels = temp_dataset.unified_labels
    
    num_classes = len(class_to_idx)

    if len(all_image_paths) < 2:
        print("Warning: Not enough data for splits. Using all data for training.")
        train_dataset = PlantImageDataset(all_image_paths, class_to_idx, transform=TRAIN_TRANSFORM)
        val_dataset = PlantImageDataset([], class_to_idx, transform=VAL_TRANSFORM) # Empty
    else:
        indices = list(range(len(all_image_paths)))
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
                num_client_data = int(len(all_image_paths) * (1-test_split_ratio))
                num_val_client = int(num_client_data * client_val_ratio_adjusted)
                val_indices = indices[:num_val_client]
                train_indices = indices[num_val_client:num_client_data]
        else:
            print("Warning: Dataset has < 2 samples per class. Using random split.")
            num_client_data = int(len(all_image_paths) * (1-test_split_ratio))
            num_val_client = int(num_client_data * client_val_ratio_adjusted)
            val_indices = indices[:num_val_client]
            train_indices = indices[num_val_client:num_client_data]

        train_paths = [all_image_paths[i] for i in train_indices]
        val_paths = [all_image_paths[i] for i in val_indices]
        
        train_dataset = PlantImageDataset(train_paths, class_to_idx, transform=TRAIN_TRANSFORM)
        val_dataset = PlantImageDataset(val_paths, class_to_idx, transform=VAL_TRANSFORM)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=robust_collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=robust_collate_fn, num_workers=2)
    
    print(f"Created train loader with {len(train_dataset)} images and val loader with {len(val_dataset)} images.")
    return train_loader, val_loader, num_classes


# --- Global Test Loader (for Server) ---
def get_global_test_loader(root_path, class_to_idx, test_split_ratio=0.2):
    """
    Loads a portion of RAW IMAGES from ALL farms to create a global test set.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    root_path = Path(root_path)
    all_image_paths = []
    all_labels = []

    print("--- Loading data from ALL farms for GLOBAL test set ---")
    farm_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("Farm_")]
    for farm_dir in farm_dirs:
        farm_paths = [p for p in farm_dir.glob('*/*.*') if p.parent.name != 'processed']
        print(f"Loading {len(farm_paths)} images from {farm_dir.name}...")
        
        temp_dataset = PlantImageDataset(farm_paths, class_to_idx, transform=None)
        all_image_paths.extend(farm_paths)
        all_labels.extend(temp_dataset.unified_labels)

    if not all_image_paths:
        raise FileNotFoundError("No processed graph files found in any farm directory.")

    print(f"Loaded a total of {len(all_image_paths)} valid images from all farms.")

    indices = list(range(len(all_image_paths)))
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    min_samples = counts.min() if len(counts) > 0 else 0

    if min_samples >= 2 and len(unique_labels) > 1:
         try:
             _, test_indices = train_test_split(
                 indices, test_size=test_split_ratio, stratify=all_labels, random_state=42
             )
         except ValueError:
             print("Warning: Stratification failed for global test set. Using random split.")
             num_test = int(len(all_image_paths) * test_split_ratio)
             test_indices = indices[-num_test:]
    else:
         print("Warning: Not enough samples for stratification. Using random split for test set.")
         num_test = int(len(all_image_paths) * test_split_ratio)
         test_indices = indices[-num_test:]

    test_paths = [all_image_paths[i] for i in test_indices]
    test_dataset = PlantImageDataset(test_paths, class_to_idx, transform=VAL_TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=robust_collate_fn)
    
    print(f"Created global test set with {len(test_dataset)} images.")
    return test_loader, len(class_to_idx)