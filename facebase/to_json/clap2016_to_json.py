from collections import defaultdict, Counter
from tqdm import tqdm
from typing import Hashable
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Define root to the dataset, the path to the output file and the number of desired folders
root = '/local/Faces'
CLAP_path = 'ChaLearn2016'
CLAP_root = os.path.join(root, CLAP_path)
output_file = '../benchmarks/databases/CLAP2016.json'

train_annotations_path = os.path.join(CLAP_root, 'train_gt.csv')
val_annotations_path = os.path.join(CLAP_root, 'valid_gt.csv')
test_annotations_path = os.path.join(CLAP_root, 'test_gt.csv')

# Annotations are defined in a matlab file
# The code the load the matlab file is 'not pretty' in python
print("Reading annotations files ...")

df_train = pd.read_csv(train_annotations_path)
df_val = pd.read_csv(val_annotations_path)
df_test = pd.read_csv(test_annotations_path)

df_train["folder"] = 0
df_val["folder"] = 1
df_test["folder"] = 2

# Prepend dataset path to file paths
df_train.image = [os.path.join(CLAP_path, "train", _) for _ in df_train.image]
df_val.image = [os.path.join(CLAP_path, "valid", _) for _ in df_val.image]
df_test.image = [os.path.join(CLAP_path, "test", _) for _ in df_test.image]

df = pd.concat([df_train, df_val, df_test], ignore_index=False, axis=0)

df = df.rename(columns={'image': 'img_path', 'mean': 'age', 'stdv': 'std_age'})

df.age = np.rint(df.age).astype(int)

df["database"] = "ChaLearn2016"

def plot_folder_hist(df, attribute, ax, normalize=False):
    df = df[df.folder != -1]
    column = df[attribute]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for folder in sorted(df.folder.unique()):
        unique_values = sorted(column.unique())
        value_counts = [sum(column[df.folder == folder] == value)
                        for value in unique_values]

        if normalize:
            value_counts = [value / (sum(value_counts) + 1e-10)
                            for value in value_counts]

        if len(unique_values) == 1:
            unique_values = [0, unique_values[0], 2]
            value_counts = [0, value_counts[0], 0]

        ax.step(unique_values, value_counts, alpha=0.5,
                color=colors[int(folder)], where='mid', label=int(folder))

    ax.legend(title="Folder")
    ax.set_xlabel(attribute.capitalize())
    ax.set_ylabel("Count")
    ax.set_title(attribute.upper())

    if len(''.join([str(_) for _ in ax.get_xticklabels()])) > 200:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    if len(''.join([str(_) for _ in ax.get_xticklabels()])) > 1000:
        ax.set_xticks([])


# Plot folder distribution fo attributes
attributes = set()
# Do not plot fields with too many or too long unique entries, e.g., bbox dictionaries
for attribute in list(sorted(df.columns)):
    if (all([isinstance(_, Hashable) for _ in df[attribute]])) and (len(df[attribute].unique()) < 501):
        attributes.add(attribute)

print("Plotting distribution of attributes in folders ...")
fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax)
plt.tight_layout()
plt.savefig("ChaLearn2016-splitting.png")

fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax, normalize=True)
plt.tight_layout()
plt.savefig("ChaLearn2016-splitting-normalized.png")

# Dump the dataframe as a list of dict to JSON
f = open(output_file, "w+")
f.write(json.dumps(df.to_dict('records'), indent=4))

f.close()
print(f"ChaLearn2016 saved to {output_file}")
