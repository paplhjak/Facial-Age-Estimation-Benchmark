import yaml
import subprocess
import pandas as pd
import os
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


if __name__ == "__main__":
    start_epoch = 1
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

        # Update data means and sigmas
        update_csv_files()