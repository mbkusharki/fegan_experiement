import streamlit as st
import subprocess
import os
import json
import time
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision import models, transforms
from torch_geometric.loader import DataLoader as GraphLoader

# Try to import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from models import FeGAN
    from preprocess import image_to_rich_graph
except ImportError as e:
    st.error(f"Could not import FeGAN modules: {e}")

# --- Constants ---
ROOT_PATH = Path("C:/Users/kusha/FeGAN_Project/data/main_data")
SERVER_LOG = "server_log.txt"
CLIENT_LOGS = "client_log_{}.txt"

st.set_page_config(page_title="FeGAN Hub", layout="wide", page_icon="🍃")

st.title("🍃 FeGAN Federated Learning Hub")
st.markdown("A unified dashboard for managing Federated Learning and running plant disease inference.")

# --- Helper Functions ---
@st.cache_data
def load_class_map():
    class_map_path = ROOT_PATH / "class_map.json"
    if class_map_path.exists():
        with open(class_map_path, 'r') as f:
            class_to_idx = json.load(f)
        return class_to_idx, {v: k for k, v in class_to_idx.items()}
    return None, None

@st.cache_resource
def load_feature_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.EfficientNet_B0_Weights.DEFAULT
    feature_extractor = models.efficientnet_b0(weights=weights)
    feature_extractor.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Identity()
    )
    modules = list(feature_extractor.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules)
    feature_extractor.add_module("flatten", torch.nn.Flatten())
    feature_extractor.eval().to(device)
    transform = weights.transforms()
    return feature_extractor, transform, device

def run_subprocess(cmd, log_file):
    # Route stdout and stderr to the log_file using shell redirection
    full_cmd = f"{cmd} > {log_file} 2>&1"
    process = subprocess.Popen(full_cmd, shell=True)
    return process

# --- Layout ---
tab1, tab2 = st.tabs(["📊 FL Dashboard (Researchers)", "🔍 Inference (Farmers)"])

# --------------------------
# TAB 1: FL DASHBOARD
# --------------------------
with tab1:
    st.header("Federated Learning Orchestrator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        mu = st.slider("Proximal Mu (FedProx)", 0.0, 1.0, 0.1, 0.01)
        epochs = st.slider("Local Epochs (Clients)", 1, 50, 10)
        sparsity = st.slider("Sparsity Level", 0.0, 0.99, 0.90, 0.01)
        
        # Discover farms
        farms = []
        if ROOT_PATH.exists():
            farms = [d.name for d in ROOT_PATH.iterdir() if d.is_dir() and d.name.startswith("Farm_")]
        
        selected_farms = st.multiselect("Select Farms to Participate", farms, default=farms)
        
        st.divider()
        st.subheader("Controls")
        
        # Start Server
        if st.button("🚀 Start Server", use_container_width=True):
            st.session_state.server_process = run_subprocess(f'"{sys.executable}" server.py --mu {mu}', SERVER_LOG)
            st.success("Server started in background.")
            
        # Start Clients
        if st.button("🌿 Start Selected Clients", use_container_width=True):
            if 'client_processes' not in st.session_state:
                st.session_state.client_processes = {}
            for farm in selected_farms:
                cmd = f'"{sys.executable}" client.py --farm-name {farm} --epochs {epochs} --sparsity {sparsity}'
                log_file = CLIENT_LOGS.format(farm)
                st.session_state.client_processes[farm] = run_subprocess(cmd, log_file)
            st.success(f"Started {len(selected_farms)} clients.")
            
    with col2:
        st.subheader("Live Logs")
        log_view = st.selectbox("View Log For:", ["Server"] + selected_farms)
        
        log_content = ""
        log_file_to_read = SERVER_LOG if log_view == "Server" else CLIENT_LOGS.format(log_view)
        
        if os.path.exists(log_file_to_read):
            with open(log_file_to_read, "r") as f:
                # Read last 30 lines
                lines = f.readlines()
                log_content = "".join(lines[-30:])
        else:
            log_content = f"Log file {log_file_to_read} not found yet."
            
        st.code(log_content, language="bash")
        
        st.button("🔄 Refresh Logs")

        st.divider()
        st.subheader("Visualizations")
        # Find images in current directory
        images = list(Path(".").glob("*.png"))
        if images:
            img_choice = st.selectbox("Select Training Graph to View", [i.name for i in images])
            st.image(str(Path(".") / img_choice))
        else:
            st.info("No visualizations found. Run a training cycle first.")

# --------------------------
# TAB 2: INFERENCE
# --------------------------
with tab2:
    st.header("Crop Disease Diagnosis")
    st.write("Upload an image of a leaf to get a diagnosis using the globally trained FeGAN model.")
    
    class_to_idx, idx_to_class = load_class_map()
    if not class_to_idx:
        st.warning("Class map not found. The model might not have been trained yet.")
    else:
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            
            # Find the best model file
            model_files = list(Path(".").glob("best_fegan_model_mu_*.pth"))
            if not model_files:
                st.error("No trained model found! Please train the model first.")
            else:
                # Pick the latest or let user select, here we just pick the first one
                model_path = model_files[0]
                st.info(f"Using model: {model_path.name}")
                
                if st.button("🔍 Diagnose", type="primary"):
                    with st.spinner("Processing image (Superpixels + EfficientNet)..."):
                        # Save temporarily
                        temp_path = "temp_uploaded_img.jpg"
                        image.convert('RGB').save(temp_path)
                        
                        feature_extractor, transform, device = load_feature_extractor()
                        
                        # Process image to graph
                        graph = image_to_rich_graph(Path(temp_path), 0, feature_extractor, transform, device)
                        
                        if graph is None:
                            st.error("Failed to generate graph from image.")
                        else:
                            st.success("Graph generated successfully!")
                            
                            with st.spinner("Running through FeGAN..."):
                                num_classes = len(class_to_idx)
                                model = FeGAN(in_channels=1280, hidden_channels=512, out_channels=num_classes).to(device)
                                
                                # Load weights
                                loaded_params_nd = torch.load(model_path, weights_only=False)
                                params_dict = zip(model.state_dict().keys(), loaded_params_nd)
                                state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
                                model.load_state_dict(state_dict, strict=True)
                                model.eval()
                                
                                # Inference
                                graph = graph.to(device)
                                # graph needs batch parameter, simulating a DataLoader batch of 1
                                from torch_geometric.data import Batch
                                batch = Batch.from_data_list([graph]).to(device)
                                
                                with torch.no_grad():
                                    outputs = model(batch)
                                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                                    confidence, predicted = torch.max(probabilities.data, 1)
                                    
                                pred_class = idx_to_class[predicted.item()]
                                conf_val = confidence.item() * 100
                                
                                st.markdown("### Diagnosis Result")
                                if "healthy" in pred_class.lower():
                                    st.success(f"**{pred_class}** ({conf_val:.2f}% confidence)")
                                else:
                                    st.error(f"**{pred_class}** ({conf_val:.2f}% confidence)")
                                    
                                st.progress(int(conf_val))
                                
                                # Show all probabilities
                                st.write("Class Probabilities:")
                                probs = probabilities.cpu().numpy()[0]
                                prob_dict = {idx_to_class[i]: p for i, p in enumerate(probs)}
                                st.bar_chart(prob_dict)
                                
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
