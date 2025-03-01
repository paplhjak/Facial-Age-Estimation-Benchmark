import os
import sys
import yaml
import json
import torch
import logging
import argparse
import subprocess
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from noise_correction.methods import *

os.environ['MKL_THREADING_LAYER'] = 'GNU'

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(output_dir):
    """Set up a separate logger for each output directory."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "correction.log")
    
    # Create a unique logger for this directory
    logger = logging.getLogger(output_dir)  # Use output_dir as the logger name
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler for logging
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    logger.addHandler(file_handler)
    logger.propagate = False  # Prevent logs from appearing in root logger

    return logger


def update_num_epochs(yaml_file_path, new_epoch):
    # Read the YAML file as plain text
    with open(yaml_file_path, 'r') as file:
        lines = file.readlines()

    # Flag to track if the num_epochs key is found
    found = False

    # Iterate through the lines to find and update num_epochs
    for i, line in enumerate(lines):
        if 'num_epochs:' in line:
            # Update the line with the new epoch value
            lines[i] = f"  num_epochs: {new_epoch}\n"
            found = True
            break  # Exit after updating

    # Raise an error if num_epochs is not found
    if not found:
        raise KeyError("The 'num_epochs' key does not exist in the YAML file.")

    # Write the updated lines back to the YAML file
    with open(yaml_file_path, 'w') as file:
        file.writelines(lines)

    print(f"Updated num_epochs to {new_epoch} in {yaml_file_path}")


def update_csv_files():
    """
    This function reads CSV files and replaces the mean and sigma values for rows
    where the folder value is 1 in one file and 0 in other files.
    """
    file_names = [f'facebase/data/Adience_256x256_resnet50_imagenet_noisy_dldl_v2/data_split{i}.csv' for i in range(5)]
    # Read all CSV files and store them in a list
    data_frames = [pd.read_csv(file, header=None) for file in file_names]

    # Loop through each file
    for i, df in enumerate(data_frames):
        # Find rows where folder == 1 in the current file
        folder_1_indices = df[df[2] == 1].index

        # Loop to replace mean and sigma values in other files
        for j, other_df in enumerate(data_frames):
            if i != j:  # Avoid comparing the file with itself
                # Find rows in other_df where folder == 0 and their indices match folder_1_indices
                update_indices = other_df[(other_df[2] == 0) & (other_df.index.isin(folder_1_indices))].index

                # Check if update_indices is not empty
                if not update_indices.empty:
                    # Replace mean and sigma values for these rows
                    other_df.loc[update_indices, 4] = df.loc[update_indices, 4]  # mean
                    other_df.loc[update_indices, 5] = df.loc[update_indices, 5]  # sigma
                else:
                    print(f"No rows to update in {file_names[j]} for folder_1_indices from {file_names[i]}.")

    # Save the updated files
    for i, df in enumerate(data_frames):
        df.to_csv(file_names[i], index=False, header=False)

    print("Files updated successfully.")


def update_means_sigmas(means, sigmas, noisy_labels, pred_labels, posteriors, update_function, config, epoch, pred_hist):
    """Applies a given update function to means and sigmas class-wise."""
    means = torch.tensor(means, dtype=torch.float32)
    sigmas = torch.tensor(sigmas, dtype=torch.float32)
    new_means = means.clone()
    new_sigmas = sigmas.clone()
    rounded_means = torch.round(means.clone()).to(dtype=torch.int)
    
    for label in torch.unique(rounded_means):
        label_mask = rounded_means == label  # Select only rows of the current class
        
        if int(label) not in pred_hist:
            pred_hist[int(label)] = list()
        pred_hist[int(label)].append((epoch,
                                 pred_labels.mean().item(),
                                 pred_labels.std().item()))
        
        if update_function == stable_mean_carl_update_function or update_function == stable_median_carl_update_function:
            new_means[label_mask], new_sigmas[label_mask] = update_function(means[label_mask],
                                                                            sigmas[label_mask],
                                                                            pred_labels[label_mask],
                                                                            posteriors[label_mask],
                                                                            config, means.clone(), label_mask)
        elif update_function == flexible_confidence_adaptive_update_function:
            new_means[label_mask], new_sigmas[label_mask] = update_function(means[label_mask],
                                                                            sigmas[label_mask],
                                                                            pred_labels[label_mask],
                                                                            posteriors[label_mask],
                                                                            config, noisy_labels,
                                                                            label_mask, rounded_means)
        else:    
            new_means[label_mask], new_sigmas[label_mask] = update_function(means[label_mask],
                                                                            sigmas[label_mask],
                                                                            pred_labels[label_mask],
                                                                            posteriors[label_mask],
                                                                            config, noisy_labels, label_mask)
    
    return new_means, new_sigmas


def update_parameters(df, ev, config, epoch, pred_hist, update_function):
    # Select only validation data where folder == 1
    validation_mask = df[2] == 1
    df_valid = df[validation_mask].copy()
    valid_indices = df_valid.index.to_numpy()  # Get original indices of validation rows
    
    # Extract required tensors from the dataframe
    noisy_labels = df_valid[3].values  # Noisy labels
    means = df_valid[4].values  # Extract as numpy array
    sigmas = df_valid[5].values  # Extract as numpy array
    pred_labels = ev['predicted_label']['age'][valid_indices]
    posteriors = ev['posterior']['age'][valid_indices]

    # Perform updates using the selected function
    new_means, new_sigmas = update_means_sigmas(means, sigmas, noisy_labels,
                                                pred_labels, posteriors, update_function,
                                                config, epoch, pred_hist)
    
    # Update dataframe only for validation entries
    df.loc[valid_indices, 4] = new_means.numpy()
    df.loc[valid_indices, 5] = new_sigmas.numpy()
    
    return df


def correct_val_labels(logger, cor_config, trn_config_name, epoch, predicted_label_history):
    data_files = [f'facebase/data/{trn_config_name}/data_split{i}.csv' for i in range(5)]
    eval_files = [f'facebase/results/{trn_config_name}/split{i}/evaluation.pt' for i in range(5)]
    config_path = f"facebase/configs/other/{trn_config_name}.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    for split, (data_file, eval_file, pred_hist) in enumerate(zip(data_files, eval_files, predicted_label_history)):
        df = pd.read_csv(data_file, header=None)
        ev = torch.load(eval_file)
        logger[split].info(f"Updating parameters of split {split} using {cor_config['correction']['method']} update function started.")
        
        updated_df = update_parameters(df, ev, config, epoch, pred_hist,
                                       update_function=globals()[cor_config['correction']['method']])
        
        # Save the updated dataframe back to CSV
        updated_df.to_csv(data_file, index=False, header=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train model with label noise correction")
    parser.add_argument("--config", type=str, default="config.yaml", required=True)
    args = parser.parse_args()

    cor_config = load_config(args.config)
    
    start_epoch = cor_config['training']['start_epoch']
    end_epoch = cor_config['training']['end_epoch']
    num_splits = cor_config['training']['num_splits']
    trn_config_name = cor_config['training']['config']

    predicted_label_history_path = [f"facebase/results/{trn_config_name}/split{split}/predicted_label_history.json" for split in range(5)]
    predicted_label_history = [dict() for split in range(5)]
    config_path = f"facebase/configs/other/{trn_config_name}.yaml"
    base_command = "python train.py {config_path} {split} --wandb-disabled"

    logger = {split: setup_logging(f"facebase/results/{trn_config_name}/split{split}") for split in range(5)}
    for split in range(5):
        logger[split].info(cor_config)

    for epoch in range(start_epoch, end_epoch + 1):
        # Initialize epoch of the config file
        update_num_epochs(config_path, epoch)
        
        for split in range(num_splits):

            logger[split].info(f"Starting processing for split {split}")
            
            # Run the command
            command = base_command.format(config_path=config_path, split=split)
            print(f"Running command: {command}")
            logger[split].info(f"Running command: {command}")
            print()

            # Execute the command and stream output to the console
            process = subprocess.run(command, shell=True)
            
            # Check if the command was successful
            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}: {command}")
                logger[split].error(f"Command failed with return code {process.returncode}: {command}")
                break

        correct_val_labels(logger, cor_config, trn_config_name, epoch, predicted_label_history)
        update_csv_files()
    
        for split in range(5):
            with open(predicted_label_history_path[split], 'w') as f:
                json.dump(predicted_label_history[split], f, indent=4)