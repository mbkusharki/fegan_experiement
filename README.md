FeGAN: A Federated Graph Attention Network for Crop Disease DetectionThis repository contains the complete Python source code for the FeGAN framework, a novel federated learning (FL) system for privacy-preserving, high-performance crop disease detection. This work is the implementation of the PhD thesis titled "FEDERATED AIoT FRAMEWORK FOR PRIVACY-PRESERVING AND SUSTAINABLE CROP DISEASE DETECTION IN PRECISION AGRICULTURE".The framework is built on a "blind server" architecture, where the server has zero knowledge of client data, and a novel "intra-image" graph conversion pipeline. This pipeline transforms each raw image into a data-rich graph, allowing a Graph Attention Network (GAT) to learn complex spatial patterns of diseases.Key FeaturesNovel Architecture: Implements an "intra-image" graph methodology. Each image is converted into a unique graph of superpixels, allowing a GAT to analyze internal spatial patterns.Rich Feature Extraction: Uses a pre-trained EfficientNet-B0 as a feature extractor to encode each graph node with a powerful 1280-dimension feature vector.Privacy-by-Design: A "Blind Server" (server.py) coordinates training without any access to client data, files, or even file paths.Handles Non-IID Data: Manages heterogeneous data from different farms (e.g., Maize, Rice, Wheat) using a universal class_map.json that intelligently unifies disparate "healthy" classes.Advanced Federated Strategy: Uses a custom AdaptiveFedProx strategy that dynamically adjusts the learning rate and allows for proximal_mu tuning to balance specialization and generalization.Communication Efficient: Implements tunable sparse updates, allowing clients (e.g., in rural areas) to send only the top 10% (or any percent) of their learned changes, drastically reducing bandwidth requirements.Full Baseline Comparison: Includes all scripts to benchmark FeGAN against standard FL-CNN, FL-Transformer, and FL-LSTM models in the same federated environment.System ArchitectureThe framework consists of two main stages:Offline Preprocessing (One-Time Cost):A farmer (client) runs preprocess.py once on their local machine.This script scans their raw image folders (e.g., Farm_1/Maize_Blight/...).It creates the universal class_map.json (if it doesn't exist) by scanning all Farm_ directories.It converts each image into a graph, extracts rich features using EfficientNet-B0, and saves the result as a small .pt file (e.g., leaf_001.pt).This heavy computation is done only once, making live training lightweight.Federated Training (Live):The "blind" server.py is started. It only reads class_map.json to know the problem shape.Each farm runs its client.py. The client loads its private, pre-processed .pt graph files.The client trains the FeGAN model locally on these graphs for a set number of epochs.The client calculates a sparse update (e.g., 90% sparsity) and sends it to the server.The server aggregates updates using AdaptiveFedProx, saves the best model, and sends the new global model back to the clients.File Structure.
├── server.py               # The "Blind" Federated Server (used for ALL experiments)
├── preprocess.py           # (STEP 1) One-time script to convert images to graphs
├── clear_cache.py          # Utility to delete all processed data and start fresh
│
├── (FeGAN - Main Model)
│   ├── client.py           # The main FeGAN client
│   ├── data_utils.py       # Loads pre-processed GRAPHS for FeGAN
│   └── models.py           # Defines the FeGAN (GAT) model
│
├── (Baseline - FL-CNN)
│   ├── client_cnn.py       # The FL-CNN client
│   ├── data_utils_cnn.py   # Loads raw IMAGES for CNN/Transformer
│   └── models_cnn.py       # Defines the EfficientNet-B0 baseline model
│
├── (Baseline - FL-Transformer)
│   ├── client_transformer.py # The FL-Transformer client
│   └── models_transformer.py # Defines the ViT-B-16 baseline model
│
└── (Baseline - FL-LSTM)
    ├── client_lstm.py      # The FL-LSTM client
    ├── data_utils_lstm.py  # Loads graph features as SEQUENCES
    └── models_lstm.py      # Defines the LSTM baseline model
Setup and InstallationThis project requires a specific set of Python libraries. It is highly recommended to use a virtual environment.# 1. Install PyTorch (CPU or GPU version, visit pytorch.org for correct command)
# Example for CPU-only:
pip install torch torchvision torchaudio

# 2. Install PyTorch Geometric (PyG)
# This is a multi-step process.
pip install torch_geometric
# Install the external binaries (CPU example):
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.0.0+cpu](https://data.pyg.org/whl/torch-2.0.0+cpu)

# 3. Install Flower (flwr)
pip install "flwr[simulation]"

# 4. Install other dependencies
pip install opencv-python-headless scikit-image scikit-learn seaborn matplotlib tqdm pandas
How to Run the ExperimentsFollow this 3-step workflow to replicate the thesis results.Step 1: Data SetupCreate a root data directory (e.g., C:/Users/kusha/FeGAN_Project/data/main_data).Inside main_data, create a folder for each farm (e.g., Farm_1, Farm_2, Farm_3).Inside each farm folder, create folders for each disease class (e.g., Maize_Blight, Rice_Blast, Healthy).Place your raw image files (.jpg, .png) inside the corresponding disease folders.Step 2: One-Time Offline PreprocessingThis step must be run before any client can be started. It generates the class_map.json and converts all images to .pt graph files.Run this script once for each farm.# In your terminal:
python preprocess.py --farm-name "Farm_1"
python preprocess.py --farm-name "Farm_2"
python preprocess.py --farm-name "Farm_3"
This will create a processed sub-folder in each farm's directory. This step can take a long time and is best run on a GPU (the script auto-detects cuda).To start a fresh experiment, run python clear_cache.py to delete all .pt files and the class_map.json.Step 3: Run the Federated ExperimentYou must run the server and all clients concurrently in separate terminals.Experiment 1: The FeGAN Framework (Main Result)Terminal 1 (Start Server):You can tune the mu value. A lower mu (e.g., 0.01) was found to be effective.python server.py --mu 0.01
Terminal 2 (Start Client 1):You can tune local epochs and sparsity.python client.py --farm-name "Farm_1" --epochs 10 --sparsity 0.9
Terminal 3 (Start Client 2):python client.py --farm-name "Farm_2" --epochs 10 --sparsity 0.9
Terminal 4 (Start Client 3):python client.py --farm-name "Farm_3" --epochs 10 --sparsity 0.9
Let the experiment run. The server will log aggregated metrics, and the best model (best_fegan_model_mu_0.01.pth), accuracy plot (validation_accuracy_mu_0.01.png), and confusion matrix (confusion_matrix_mu_0.01.png) will be saved in your project folder.Experiment 2: The FL-CNN Baseline (for Comparison)Terminal 1 (Start Server):You can use the same server script with the same or different mu.python server.py --mu 0.1
Terminal 2 (Start Client 1):Run the client_cnn.py script. Note that --sparsity is not used here.python client_cnn.py --farm-name "Farm_1" --epochs 10
Terminal 3 (Start Client 2):python client_cnn.py --farm-name "Farm_2" --epochs 10
Terminal 4 (Start Client 3):python client_cnn.py --farm-name "Farm_3" --epochs 10
Repeat this process for client_transformer.py and client_lstm.py to generate all baseline results for your comparison in Chapter 4.ResultsThis framework was successfully validated, achieving a final global accuracy of 90.47% (F1-Score: 81.00%). The FeGAN model demonstrably outperformed all baselines, including FL-Transformer (88.57% Acc), FL-CNN (87.94% Acc), and FL-LSTM (84.63% Acc), confirming the superiority of the "intra-image" graph methodology.
