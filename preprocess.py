# -*- coding: utf-8 -*-
"""
preprocess.py

This is the definitive, high-performance preprocessing script.
Its responsibilities are:
1. Create a universal class map if it doesn't exist.
2. For a specific farm, find all raw images.
3. Check which images have already been processed into .pt files.
4. Only process the new, unprocessed images into individual graph files
   using the powerful EfficientNet-B0 feature extractor.
5. Save these new graph files into the correct 'processed' subdirectory.
"""
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import torch
import cv2
from skimage.segmentation import slic
import numpy as np
import torch_geometric.data
from torchvision import models, transforms
from PIL import Image
import warnings

# --- Definitive Graph Creation Function (with Rich Features & Re-indexing) ---
def image_to_rich_graph(img_path, class_idx, feature_extractor, transform, device, n_segments=50, resize_dim=(224, 224)):
    """Converts a single image into a robust graph object with rich features."""
    try:
        image_for_slic = cv2.imread(str(img_path))
        if image_for_slic is None: return None
        image_for_slic = cv2.resize(image_for_slic, resize_dim)
        image_for_slic = cv2.cvtColor(image_for_slic, cv2.COLOR_BGR2RGB)

        pil_image = Image.open(img_path).convert('RGB').resize(resize_dim)

        # Generate superpixels with start_label=0
        segments = slic(image_for_slic, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
        unique_segments = np.unique(segments)

        # Check if segments start from 0, otherwise remap (handles potential skimage variations)
        if unique_segments[0] != 0 or not np.all(np.diff(unique_segments) == 1):
             segment_map = {old_id: new_id for new_id, old_id in enumerate(unique_segments)}
             remapped_segments = np.vectorize(segment_map.get)(segments)
        else:
            remapped_segments = segments # Already 0-indexed and contiguous


        node_features = []
        with torch.no_grad():
            for seg_val in np.unique(segments): # Iterate using original segment values for masking
                mask = (segments == seg_val)
                # Ensure mask is boolean for correct multiplication
                mask_bool = mask.astype(bool)
                masked_image_np = np.array(pil_image) * np.expand_dims(mask_bool, axis=-1)
                masked_image_pil = Image.fromarray(masked_image_np.astype('uint8'), 'RGB')

                input_tensor = transform(masked_image_pil).unsqueeze(0).to(device)
                features = feature_extractor(input_tensor).squeeze().cpu()
                node_features.append(features)

        # Ensure number of features matches the number of unique segments found
        if len(node_features) != len(np.unique(remapped_segments)):
             print(f"Warning: Mismatch in node features ({len(node_features)}) and segments ({len(np.unique(remapped_segments))}) for {img_path}. Skipping.")
             return None

        x = torch.stack(node_features)


        edges = []
        # Create edges based on adjacency in the remapped segments
        for i in range(remapped_segments.shape[0]):
            for j in range(remapped_segments.shape[1]):
                current_node = remapped_segments[i, j]
                # Check right neighbor
                if j + 1 < remapped_segments.shape[1]:
                    right_node = remapped_segments[i, j + 1]
                    if current_node != right_node:
                        edges.append(sorted((current_node, right_node)))
                # Check bottom neighbor
                if i + 1 < remapped_segments.shape[0]:
                    bottom_node = remapped_segments[i + 1, j]
                    if current_node != bottom_node:
                        edges.append(sorted((current_node, bottom_node)))

        if not edges: return None

        unique_edges = np.unique(np.array(edges), axis=0)
        edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()

        # Final check for edge index validity
        num_nodes = x.shape[0]
        if edge_index.max().item() >= num_nodes:
            print(f"Warning: Invalid edge index found (max index {edge_index.max().item()} >= num_nodes {num_nodes}) for {img_path}. Skipping.")
            return None


        graph = torch_geometric.data.Data(x=x, edge_index=edge_index, y=torch.tensor([class_idx], dtype=torch.long))
        return graph
    except Exception as e:
        print(f"Warning: Skipping image {img_path} due to error during graph creation: {e}")
        return None

# --- Main Preprocessing Execution Block ---
def main():
    warnings.filterwarnings("ignore", category=UserWarning) # Suppress common warnings
    parser = argparse.ArgumentParser(description="FeGAN Smart Preprocessing Utility")
    parser.add_argument("--farm-name", type=str, required=True, help="e.g., Farm_1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root_path = Path("C:/Users/kusha/FeGAN_Project/data/main_data")
    farm_path = root_path / args.farm_name
    processed_path = farm_path / "processed"
    processed_path.mkdir(exist_ok=True)

    print("Loading pre-trained feature extractor (EfficientNet-B0)...")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    feature_extractor = models.efficientnet_b0(weights=weights)
    # Correctly remove the classifier head
    feature_extractor.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), # Default dropout for efficientnet_b0
        torch.nn.Identity() # Output features directly
    )
    # Adjust feature extractor to directly output features after the pooling layer
    # EfficientNet's features are typically extracted before the final classifier layers.
    # The avgpool layer outputs the features we need.
    modules = list(feature_extractor.children())[:-1] # Remove the original classifier Sequential block
    feature_extractor = torch.nn.Sequential(*modules)
    # Add a flatten layer to get a 1D feature vector per image patch
    feature_extractor.add_module("flatten", torch.nn.Flatten())


    feature_extractor.eval()
    feature_extractor.to(device)
    transform = weights.transforms()

    class_map_path = root_path / "class_map.json"
    if not class_map_path.exists():
        print("Universal class map not found. Creating it for the first time...")
        class_names = set()
        for fp_dir in root_path.iterdir(): # Use fp_dir instead of farm_dir
            if fp_dir.is_dir() and fp_dir.name.startswith("Farm_"):
                for class_dir in fp_dir.iterdir():
                    if class_dir.is_dir() and class_dir.name != "processed":
                        class_name = class_dir.name
                        if "healthy" in class_name.lower():
                            class_names.add("Healthy")
                        else:
                            class_names.add(class_name)
        class_to_idx = {name: i for i, name in enumerate(sorted(list(class_names)))}
        with open(class_map_path, 'w') as f:
            json.dump(class_to_idx, f, indent=4)
        print(f"Created and saved universal class map with {len(class_to_idx)} classes.")
    else:
        with open(class_map_path, 'r') as f:
            class_to_idx = json.load(f)
        print(f"Loaded existing universal class map with {len(class_to_idx)} classes.")


    print(f"\n--- Starting Smart Preprocessing for {args.farm_name} ---")
    raw_image_paths = list(farm_path.glob('*/*.*'))
    processed_graph_names = {p.stem for p in processed_path.glob('*.pt')}
    images_to_process = [p for p in raw_image_paths if p.stem not in processed_graph_names]

    if not images_to_process:
        print("All images are already processed.")
        return

    print(f"Found {len(images_to_process)} new images to process.")
    processed_count = 0
    for img_path in tqdm(images_to_process, desc=f"Processing new images for {args.farm_name}"):
        class_name = img_path.parent.name
        unified_class_name = "Healthy" if "healthy" in class_name.lower() else class_name

        if unified_class_name in class_to_idx:
            class_idx = class_to_idx[unified_class_name]
            graph = image_to_rich_graph(img_path, class_idx, feature_extractor, transform, device)

            if graph:
                save_path = processed_path / f"{img_path.stem}.pt"
                torch.save(graph, save_path)
                processed_count += 1

    print(f"\nSuccessfully processed and saved {processed_count} new graphs.")
    print("--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()

