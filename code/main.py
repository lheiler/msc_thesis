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
from Visualization import tsne # Assuming you have a tsne visualization function

_FLOAT64_RE = re.compile(r'np\.float64\(([^)]+)\)')  # capture inner number

def load_latent_parameters_array(file_path, batch_size: int = 32):
    """
    Read a text file where each line is:
        ({'G_ee': np.float64(...), ...}, label, age, abn)
    and return a DataLoader that yields tuples:
        (np.ndarray[float32], label, age, abn)
    """
    latent_params = []

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


def main():
    """
    Main function to run the entire pipeline.
    """
    method = "c22"  # Specify the method for latent feature extraction, e.g., "ctm" or "c22"

    # path parameters:
    data_path_train = "/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean/train"  # Specify the path to your data
    data_path_eval = "/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean/eval"  # Specify the path to your evaluation data
    results_path = "Results/tuh-eeg-" + method + "-parameters/"  # Specify the path to save results
    
    
    
    # model parameters:
    batch_size = 32  # Specify the batch size for data loading
    num_epochs = 20  # Specify the number of epochs for training
    hidden_layer_size = 128  # Specify the size of the hidden layer in the model
    hidden_layers = 2  # Specify the number of hidden layers in the model
    
    extracted = False  # Set to True if latent features are already extracted, otherwise False
    
    
    # ------------------------    # 1. Load and preprocess data.  ------------------------
    
    if not extracted:
        # Load and preprocess data
        print("Loading data...")
        t_data = dl.load_data(data_path_train)
        e_data = dl.load_data(data_path_eval)
        # ------------------------------------------------------------------------------------
        
        
        # Extract latent features
        print("Extracting latent features...")
        t_latent_features = extractor.extract_latent_features(t_data, batch_size=batch_size, save_path=os.path.join(results_path,"temp_latent_features_train.txt"), method=method)
        e_latent_features = extractor.extract_latent_features(e_data, batch_size=batch_size, save_path=os.path.join(results_path,"temp_latent_features_eval.txt"), method=method)
        
        #np.save("Results/tuh-eeg-ctm-parameters/t_latent_features.npy", t_latent_features.dataset.tensors[0].numpy())
        #np.save("Results/tuh-eeg-ctm-parameters/e_latent_features.npy", e_latent_features.dataset.tensors[0].numpy())
        #print("Latent features extracted and saved as numpy arrays.")
    # ------------------------------------------------------------------------------------
    else:
        # Load latent features from saved files
        print("Loading latent features from saved files...")
        t_latent_features = load_latent_parameters_array(os.path.join(results_path, "temp_latent_features_train.txt"), batch_size=batch_size)
        e_latent_features = load_latent_parameters_array(os.path.join(results_path, "temp_latent_features_eval.txt"), batch_size=batch_size)
        
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
