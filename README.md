# 🍃 FeGAN: Federated Graph Attention Network for Crop Disease Classification

FeGAN is a robust, privacy-preserving Federated Learning (FL) system that leverages Graph Neural Networks (GNNs) for highly accurate crop disease diagnosis. By distributing the training process across multiple edge devices ("Farms") using the [Flower](https://flower.ai/) framework, FeGAN allows agricultural datasets to remain private while collectively training a powerful global model.

The system uses **EfficientNet-B0** combined with **SLIC Superpixels** to extract localized graph representations of leaf images, which are then processed by a customized **Graph Attention Network (GAT)**.

---

## ✨ Features

* **Federated Learning Architecture**: Powered by Flower (`flwr`), enabling decentralized training across multiple nodes.
* **Adaptive FedProx Strategy**: Custom server-side strategy with a tunable proximal term ($\mu$) to handle non-IID data distributions, featuring patience-based learning rate decay and continual learning (auto-saving/resuming).
* **Communication-Efficient Updates**: Client-side parameter sparsification (e.g., zeroing out 90% of updates) significantly reduces the network bandwidth required for federated rounds.
* **Graph-Based Vision Pipeline**: 
  * Uses SLIC to segment images into biologically meaningful superpixels.
  * Extracts deep features from each superpixel using EfficientNet-B0.
  * Connects adjacent superpixels with spatial edges to form a PyTorch Geometric graph.
* **Stratified Data Loading**: Ensures balanced train/validation/test splits even across highly skewed local farm datasets.
* **Streamlit Dashboard Hub**: A beautiful, unified web UI for both researchers (to orchestrate FL) and farmers (to run live inference).

---

## 📂 Repository Structure

* `preprocess.py`: Converts raw leaf images into rich `.pt` graph objects using EfficientNet and SLIC.
* `server.py`: The central Flower server script running the custom `AdaptiveFedProx` strategy.
* `client.py`: The edge client script that trains locally and sparsifies gradient updates.
* `models.py`: Defines the high-capacity `FeGAN` PyTorch model architecture.
* `data_utils.py`: Handles loading and stratified splitting of the `.pt` graph files.
* `app.py`: The interactive Streamlit web dashboard.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/gemini_fegan.git
   cd gemini_fegan
   ```

2. **Install the dependencies:**
   It's recommended to use a virtual environment. You will need PyTorch, PyTorch Geometric, Flower, Streamlit, and OpenCV.
   ```bash
   pip install torch torchvision
   pip install torch_geometric
   pip install flwr scikit-learn scikit-image opencv-python matplotlib seaborn streamlit
   ```

---

## 🚀 Usage

### 1. The Streamlit Hub (Recommended)
The easiest way to interact with the system is through the unified web dashboard.

```bash
streamlit run app.py
```
* **Dashboard Tab**: Select which farms to train on, configure the FedProx $\mu$, Sparsity level, and Epochs, then click "Start Server" and "Start Clients". View live terminal logs and training artifacts directly in the browser.
* **Inference Tab**: Upload an image of a leaf. The app will generate the graph on the fly, run it through the globally trained model (`best_fegan_model.pth`), and output the diagnosis with confidence scores.

### 2. Command Line Execution
If you prefer running the components manually:

**Step A: Preprocessing**
Generate the graph representations for a specific farm.
```bash
python preprocess.py --farm-name "Farm_1"
```

**Step B: Start the Server**
Start the central aggregator.
```bash
python server.py --mu 0.1
```

**Step C: Start the Clients**
Open new terminal windows for each client/farm you want to participate in the training round.
```bash
python client.py --farm-name "Farm_1" --epochs 10 --sparsity 0.9
python client.py --farm-name "Farm_2" --epochs 10 --sparsity 0.9
```

---

## 📊 Outputs & Artifacts

During training, the server automatically generates and saves:
* `best_fegan_model_mu_<val>.pth`: The highest-performing model weights.
* `validation_accuracy_mu_<val>.png`: A plot of the global validation accuracy over FL rounds.
* `confusion_matrix_mu_<val>.png`: A detailed confusion matrix evaluated on the global test set.

---
*Built with PyTorch, PyTorch Geometric, and Flower.*
