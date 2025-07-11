import faulthandler
faulthandler.enable()

from Data_Preprocessing import data_loading as dl
import Model_Training.classification_model as cm
from Model_Training.classification_model import ClassificationModel
import Evaluation.evaluation as eval
import Latent_Extraction.extractor as extractor
import numpy as np
import os
import ast
from torch.utils.data import DataLoader, TensorDataset
import re
import json
import torch
from Visualization import tsne # Assuming you have a tsne visualization function
import argparse
import yaml
from pathlib import Path

_FLOAT64_RE = re.compile(r'np\.float64\(([^)]+)\)')  # capture inner number

def load_latent_parameters_array(file_path, batch_size: int = 32):
    """
    Read a text file where each line is:
        ({'G_ee': np.float64(...), ...}, label, age, abn)
    and return a DataLoader that yields tuples:
        (np.ndarray[float32], label, age, abn)
    """
    latent_params = []
    file_path = file_path + ".txt" if not file_path.endswith(".txt") else file_path
    with open(file_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Remove np.float64(...) so literal_eval can parse safely
            cleaned = _FLOAT64_RE.sub(r'\1', line)

            # Parse tuple -> (dict, label, age, abn)
            try:
                param_dict, label, age, abn = ast.literal_eval(cleaned)
            except (ValueError, SyntaxError) as err:
                print(f"Skipping malformed line: {raw_line[:80]} … ({err})")
                continue

            # Convert dict values to a float64 NumPy vector
            param_values = np.array([float(v) for v in param_dict.values()],
                                    dtype=np.float32)

            latent_params.append((param_values, label, age, abn))
    return DataLoader(latent_params, batch_size=batch_size, shuffle=False)

def load_latent_c22_parameters_array(file_path, batch_size: int = 32):
    latent_params = []
    file_path = file_path + ".json" if not file_path.endswith(".json") else file_path
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    latent_vec = torch.tensor(entry[0], dtype=torch.float32)
                    g = torch.tensor(entry[1], dtype=torch.float32)
                    a = torch.tensor(entry[2], dtype=torch.float32)
                    ab = torch.tensor(entry[3], dtype=torch.float32)

                    latent_params.append((latent_vec, g, a, ab))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")

    print(f"Loaded {len(latent_params)} latent parameters from {file_path}")
    return DataLoader(latent_params, batch_size=batch_size, shuffle=False)
            

def load_latent_ae_parameters_array(file_path, batch_size: int = 32):
    """
    Read a text file where each line is:
        [np.ndarray[float32], label, age, abn]
    and return a DataLoader that yields tuples:
        (np.ndarray[float32], label, age, abn)
    """
    latent_params = []
    file_path = file_path + ".json" if not file_path.endswith(".json") else file_path
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    latent_vec = torch.tensor(entry[0], dtype=torch.float32)
                    g = torch.tensor(entry[1], dtype=torch.float32)
                    a = torch.tensor(entry[2], dtype=torch.float32)
                    ab = torch.tensor(entry[3], dtype=torch.float32)

                    latent_params.append((latent_vec, g, a, ab))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")

    print(f"Loaded {len(latent_params)} latent parameters from {file_path}")
    return DataLoader(latent_params, batch_size=batch_size, shuffle=False)
        
            
def main():
    """
    Main function to run the entire pipeline.
    """
    # 0️⃣ Load configuration from YAML / CLI -------------------------------
    parser = argparse.ArgumentParser(description="EEG classification pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ------------------------  Parameters from config  ------------------------
    method    = cfg.get("method", "ctm")
    data_corp = cfg.get("data_corp", "harvard")

    paths_cfg = cfg.get("paths", {})
    data_path_train   = paths_cfg.get("data_train", "")
    data_path_eval    = paths_cfg.get("data_eval", "")
    data_path_harvard = paths_cfg.get("data_harvard", "")

    results_root = paths_cfg.get("results_root", "Results")
    if data_corp == "harvard":
        results_path = os.path.join(results_root, f"harvard-eeg-{method}-parameters-100abnormal")
    else:
        results_path = os.path.join(results_root, f"tuh-eeg-{method}-parameters")
    os.makedirs(results_path, exist_ok=True)

    model_cfg = cfg.get("model", {})
    batch_size        = model_cfg.get("batch_size", 16)
    num_epochs        = model_cfg.get("num_epochs", 20)
    hidden_layer_size = model_cfg.get("hidden_layer_size", 128)
    hidden_layers     = model_cfg.get("hidden_layers", 2)

    extracted = cfg.get("extracted", False)
    
    
    # ------------------------  # 1. Load and preprocess data.  ------------------------
    
    if not extracted:
        # Load and preprocess data
        print("Loading data...")
        if data_corp == "harvard":
            t_data, e_data = dl.load_data_harvard(data_path_harvard)
        else:
            t_data = dl.load_data(data_path_train)
            e_data = dl.load_data(data_path_eval)
        # ------------------------------------------------------------------------------------
        
        
        # Extract latent features
        print("Extracting latent features...")
        t_latent_features = extractor.extract_latent_features(t_data, batch_size=batch_size, save_path=os.path.join(results_path,"temp_latent_features_train.json"), method=method)
        e_latent_features = extractor.extract_latent_features(e_data, batch_size=batch_size, save_path=os.path.join(results_path,"temp_latent_features_eval.json"), method=method)
        
        #np.save("Results/tuh-eeg-ctm-parameters/t_latent_features.npy", t_latent_features.dataset.tensors[0].numpy())
        #np.save("Results/tuh-eeg-ctm-parameters/e_latent_features.npy", e_latent_features.dataset.tensors[0].numpy())
        #print("Latent features extracted and saved as numpy arrays.")
    # ------------------------------------------------------------------------------------
    else:
        # Load latent features from saved files
        print("Loading latent features from saved files...")
        t_latent_features = load_latent_ae_parameters_array(os.path.join(results_path, "temp_latent_features_train"), batch_size=batch_size)
        e_latent_features = load_latent_ae_parameters_array(os.path.join(results_path, "temp_latent_features_eval"), batch_size=batch_size)
        
        print("Latent features loaded successfully.")
    
    # ------------------------    # visualize tsne  ------------------------
    
    # tsne.tsne_plot(t_latent_features, results_path)
    
    # ------------------------    # 2. Train the classification model.  ------------------------
    
    
    # Train the classification model
    print("Training classification model...")
    model = ClassificationModel(input_dim=t_latent_features.dataset[0][0].shape[0])
    cm.train(model, t_latent_features, n_epochs=num_epochs)
    # Evaluate the model
    print("Evaluating model...")
    evaluation_results = eval.run_evaluation(model, e_latent_features, save_path=results_path)

    
    # Save the results
    print("Saving results...")
    eval.save_results(evaluation_results, results_path)
    
    # Print completion message
    print("Pipeline completed successfully!")
    

    
if __name__ == "__main__":
    main()
