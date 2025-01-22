import os
import json
import pandas as pd
from tqdm import tqdm

# Define the root path to the dataset and the output file path
root = '/home/vision/alireza-sm/datasets/Adience'
output_file = '/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/benchmarks/databases/Adience.json'

# Define the folder containing cleaned fold text files
fold_folder = os.path.join(root, 'folds')

def convert_txt_to_json(root, fold_folder, output_file):
    """
    Convert Adience dataset's cleaned fold txt files into a single JSON file.

    Parameters
    ----------
    root : str
        Root path of the Adience dataset.
    fold_folder : str
        Path to the folder containing cleaned fold txt files.
    output_file : str
        Path to save the output JSON file.
    """
    # Initialize a list to hold all JSON objects
    all_data = []

    # Process each fold file
    for fold_file in tqdm(sorted(os.listdir(fold_folder))):
        print(fold_file)
        if not fold_file.endswith('.txt'):
            print(fold_file, "skipped.")
            continue

        # Extract folder number from the filename
        folder_number = int(fold_file.split('_')[2])
        fold_path = os.path.join(fold_folder, fold_file)

        # Load the fold data
        df = pd.read_csv(fold_path, sep='\t')

        # Iterate over each row and construct the JSON object
        for _, row in df.iterrows():
            # Construct image path
            image_path = f"Adience/aligned/{row['user_id']}/landmark_aligned_face.{row['face_id']}.{row['original_image']}"

            # Format the JSON object
            data_instance = {
                "img_path": image_path,
                "id_num": int(row['face_id']),
                "age": row['age'] if pd.notna(row['age']) else None,
                "gender": row['gender'].upper() if pd.notna(row['gender']) else None,
                "database": "Adience",
                "folder": folder_number
            }

            # Append the instance to the list
            all_data.append(data_instance)

    # Write the list of JSON objects to a file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

    print(f"Adience data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    convert_txt_to_json(root, fold_folder, output_file)
