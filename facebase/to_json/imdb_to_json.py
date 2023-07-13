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
IMDB_path = 'IMDB_WIKI/imdb'
IMDB_root = os.path.join(root, IMDB_path)
annotations_path = os.path.join('/local/Faces/IMDB_WIKI/em_annot', 'emcnn_annotion_18-Nov-2017.txt')
output_file = '../benchmarks/databases/IMDB-EM-CNN.json'
nr_folders = 10
confidence_threshold = 0.0

print("Reading annotations ...")
# Build dataframe from the annotations csv
df = pd.read_csv(annotations_path, sep='\t', names=[
                 'img_path', 'name', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'age', 'gender', 'confidence'])
# Keep only files with confidence over the threshold
df = df.loc[df['confidence'] >= confidence_threshold]

# Prepend dataset path to file paths
df.img_path = IMDB_path + os.sep + df.img_path

print("Concatenating bounding box annotations ...")
# Build bounding box annotation
df['bbox'] = [[int(__) for __ in _] for _ in np.stack([df.bbox_left, df.bbox_top, df.bbox_right,
                                         df.bbox_top, df.bbox_right, df.bbox_bottom, df.bbox_left, df.bbox_bottom]).T.astype(int)]
df.drop(['bbox_left', 'bbox_right', 'bbox_bottom',
        'bbox_top'], inplace=True, axis=1)


# Prepare list of unique identities
unique_ids = df.name.unique()
np.random.shuffle(unique_ids)

# Assign identities to folders
id2folder = defaultdict(int)

print("Splitting the files into subject exclusive folders ...")
folder = 0
for unique_id in tqdm(unique_ids):
    id2folder[unique_id] = folder
    folder += 1
    folder %= nr_folders

folders = []
for name in df.name:
    folders.append(id2folder[name])

df["folder"] = folders

df.folder = df.folder.astype(int)

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
plt.savefig("IMDB-splitting.png")

fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax, normalize=True)
plt.tight_layout()
plt.savefig("IMDB-splitting-normalized.png")

# Dump the dataframe as a list of dict to JSON
f = open(output_file, "w+")
f.write(json.dumps(df.to_dict('records'), indent=4))

f.close()
print(f"IMDB-EM-CNN saved to {output_file}")
