from collections import defaultdict, Counter
from tqdm import tqdm
from typing import Hashable
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

# Define root to the dataset, the path to the output file and the number of desired folders
root = '/local/Faces'
IMDB_path = 'IMDB_WIKI/imdb/'
IMDB_root = os.path.join(root, IMDB_path)
annotations_root = '.'

train_annotations_path = os.path.join(annotations_root, 'imdb_train_new.csv')
val_annotations_path = os.path.join(annotations_root, 'imdb_valid_new.csv')
test_annotations_path = os.path.join(annotations_root, 'imdb_test_new.csv')

output_file = '../benchmarks/databases/IMDB-CLEAN-FP-AGE.json'

print("Reading annotations ...")
# Build dataframe from the annotations csv
df_train = pd.read_csv(train_annotations_path, sep=',', names=['img_path', 'age', 'gender', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'roll', 'yaw', 'pitch'], skiprows=[0])
df_val = pd.read_csv(val_annotations_path, sep=',', names=['img_path', 'age', 'gender', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'roll', 'yaw', 'pitch'], skiprows=[0])
df_test = pd.read_csv(test_annotations_path, sep=',', names=['img_path', 'age', 'gender', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'roll', 'yaw', 'pitch'], skiprows=[0])

df_train["folder"] = 0
df_val["folder"] = 1
df_test["folder"] = 2

# Concatenate Train/Test/Val
df = pd.concat([df_train, df_val, df_test], ignore_index=False, axis=0)

# Extract person identifier from filepath
def get_name(row):
    path = row['img_path']
    match = re.search(r'[0-9]+\/nm([0-9]+)_rm', path)
    return match[1]

df['name'] = df.apply(get_name, axis=1)
df["database"] = ["IMDB-CLEAN" for _ in df.img_path]

# Prepend root path to image path
df.img_path = IMDB_path + df.img_path

print("Concatenating bounding box annotations ...")
# Build bounding box annotation
df['bbox'] = [[int(__) for __ in _] for _ in np.stack([df.bbox_left, df.bbox_top, df.bbox_right,
                                         df.bbox_top, df.bbox_right, df.bbox_bottom, df.bbox_left, df.bbox_bottom]).T.astype(int)]
df.drop(['bbox_left', 'bbox_right', 'bbox_bottom',
        'bbox_top'], inplace=True, axis=1)


print("Splitting of the dataset is not subject exclusive. Making sure that Test is subject exclusive ...")
for name in tqdm(df.name.unique()):
    if not len(df.folder[df.name == name].unique()) == 1:
        assert 2 not in df.folder[df.name==name].unique()

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
    if (all([isinstance(_, Hashable) for _ in df[attribute]])) and (len(df[attribute].unique()) < 500):
        attributes.add(attribute)

print("Plotting distribution of attributes in folders ...")
fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax)
plt.tight_layout()
plt.savefig("IMDB-CLEAN-splitting.png")

fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax, normalize=True)
plt.tight_layout()
plt.savefig("IMDB-CLEAN-splitting-normalized.png")

# Dump the dataframe as a list of dict to JSON
f = open(output_file, "w+")
f.write(json.dumps(df.to_dict('records'), indent=4))

f.close()
print(f"IMDB-FP-AGE saved to {output_file}")
