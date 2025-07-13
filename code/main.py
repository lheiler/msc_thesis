import faulthandler
faulthandler.enable()

from data_preprocessing import data_loading as dl
import model_training.classification_model as cm
from model_training.classification_model import ClassificationModel
import evaluation.evaluation as eval
import latent_extraction.extractor as extractor
import numpy as np
import os
import ast
from torch.utils.data import DataLoader
import re
import json
import torch
import argparse
import yaml
from utils.latent_loading import (
    load_latent_parameters_array,
    load_latent_c22_parameters_array,
    load_latent_ae_parameters_array,)

# Dataset stats helper
from utils.data_metrics import compute_dataset_stats

# Helper functions have been moved to utils.latent_loading

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
    batch_size    = model_cfg.get("batch_size", 16)
    num_epochs    = model_cfg.get("num_epochs", 20)
    dropout       = model_cfg.get("dropout", 0.2)
    weight_decay  = model_cfg.get("weight_decay", 0.0)
    scheduler     = model_cfg.get("scheduler", "none")

    loss_cfg = cfg.get("loss_weights", {})
    lambda_gender = loss_cfg.get("lambda_gender", 0.0)
    lambda_age    = loss_cfg.get("lambda_age", 0.0)
    lambda_abn    = loss_cfg.get("lambda_abn", 1.0)

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

    # --------------------------------------------------------------------
    # Safety check: ensure we have data before proceeding
    # --------------------------------------------------------------------
    if len(t_latent_features.dataset) == 0 or len(e_latent_features.dataset) == 0:
        msg = (
            "❌ No training or evaluation samples were loaded. "
            "Please verify the dataset paths and that preprocessing succeeded."
        )
        print(msg)
        return
    
    # ------------------------    # visualize tsne  ------------------------
    
    # tsne.tsne_plot(t_latent_features, results_path)
    
    # ------------------------    # 2. Train the classification model.  ------------------------
    
    
    # Train the classification model
    print("Training classification model...")
    model = ClassificationModel(input_dim=t_latent_features.dataset[0][0].shape[0], dropout=dropout)
    cm.train(
        model,
        t_latent_features,
        n_epochs=num_epochs,
        λ_gender=lambda_gender,
        λ_age=lambda_age,
        λ_abn=lambda_abn,
        weight_decay=weight_decay,
        scheduler=scheduler,
    )
    # ------------------------------------------------------------
    # 3. Dataset descriptive statistics
    # ------------------------------------------------------------
    train_stats = compute_dataset_stats(t_latent_features)
    eval_stats  = compute_dataset_stats(e_latent_features)

    # 4. Evaluate the model
    evaluation_results = eval.run_evaluation(model, e_latent_features, save_path=results_path)

    # 5. Merge dataset stats into metrics before saving
    evaluation_results["train_dataset_stats"] = train_stats
    evaluation_results["eval_dataset_stats"]  = eval_stats
    
    # Save the results
    print("Saving results...")
    eval.save_results(evaluation_results, results_path)
    
    # Print completion message
    print("Pipeline completed successfully!")
    

    
if __name__ == "__main__":
    main()
