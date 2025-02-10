import os
import yaml
import torch
import subprocess
import pandas as pd
from torch.nn.functional import softmax

os.environ['MKL_THREADING_LAYER'] = 'GNU'


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

def update_means_sigmas(means, sigmas, noisy_labels, ev, indices, update_function, config):
    """Applies a given update function to means and sigmas class-wise."""
    means = torch.tensor(means, dtype=torch.float32)
    sigmas = torch.tensor(sigmas, dtype=torch.float32)
    new_means = means.clone()
    new_sigmas = sigmas.clone()
    
    for label in set(noisy_labels):
        mask = noisy_labels == label  # Select only rows of the current class
        ev_indices = indices[mask]  # Map to original indices
        new_means[mask], new_sigmas[mask] = update_function(
            means[mask], sigmas[mask], ev, ev_indices, config
        )
    
    return new_means, new_sigmas

def default_update_function(means, sigmas, ev, indices, config):
    """Default update function for means and sigmas."""
    alpha = config['training']['alpha']
    beta = config['training']['beta']
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha * (error - sigmas)
    new_means = beta * means + (1 - beta) * best_predicted_labels
    return new_means, new_sigmas

def confidence_adaptive_update_function(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    # Clip confidence to a minimum of 0
    confidence = torch.clamp(confidence, min=0)
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(all_noisy_labels)
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    # Update means and sigmas
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha_k * (error - sigmas)
    new_means = (1 - beta_k) * means + beta_k * best_predicted_labels
    
    return new_means, new_sigmas

def update_parameters(df, ev, config, update_function=default_update_function):
    # Select only validation data where folder == 1
    validation_mask = df[2] == 1
    df_valid = df[validation_mask].copy()
    valid_indices = df_valid.index.to_numpy()  # Get original indices of validation rows
    
    # Extract required tensors from the dataframe
    means = df_valid[4].values  # Extract as numpy array
    sigmas = df_valid[5].values  # Extract as numpy array
    noisy_labels = df_valid[3].values  # Noisy labels
    
    # Perform updates using the selected function
    new_means, new_sigmas = update_means_sigmas(means, sigmas, noisy_labels, ev, valid_indices, update_function, config)
    
    # Update dataframe only for validation entries
    df.loc[valid_indices, 4] = new_means.numpy()
    df.loc[valid_indices, 5] = new_sigmas.numpy()
    
    return df

def correct_val_labels():
    data_files = [f'facebase/data/Adience_256x256_resnet50_imagenet_noisy_dldl_v2/data_split{i}.csv' for i in range(5)]
    eval_files = [f'facebase/results/Adience_256x256_resnet50_imagenet_noisy_dldl_v2/split{i}/evaluation.pt' for i in range(5)]
    config_path = "facebase/configs/other/Adience_256x256_resnet50_imagenet_noisy_dldl_v2.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    for data_file, eval_file in zip(data_files, eval_files):
        df = pd.read_csv(data_file, header=None)
        ev = torch.load(eval_file)
        
        updated_df = update_parameters(df, ev, config, update_function=confidence_adaptive_update_function)
        
        # Save the updated dataframe back to CSV
        updated_df.to_csv(data_file, index=False, header=False)


if __name__ == "__main__":
    start_epoch = 10
    end_epoch = 50
    num_splits = 5

    config_path = "facebase/configs/other/Adience_256x256_resnet50_imagenet_noisy_dldl_v2.yaml"
    base_command = "python train.py {config_path} {split} --wandb-disabled"

    for epoch in range(start_epoch, end_epoch + 1):
        # Initialize epoch of the config file
        update_num_epochs(config_path, epoch)
        for split in range(num_splits):
            # Run the command
            command = base_command.format(config_path=config_path, split=split)
            print(f"Running command: {command}")

            # Execute the command and stream output to the console
            process = subprocess.run(command, shell=True)
            
            # Check if the command was successful
            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}: {command}")
                break

        correct_val_labels()
        # break
        # Update data means and sigmas
        update_csv_files()