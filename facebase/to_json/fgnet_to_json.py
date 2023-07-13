from collections import defaultdict, Counter
from tqdm import tqdm
from typing import Hashable
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

# Define root to the dataset, the path to the output file and the number of desired folders
root = '/local/Faces'
FGNet_path = 'FG-NET/images'
FGNet_root = os.path.join(root, FGNet_path)
output_file = '../benchmarks/databases/FG-Net.json'

# FGNET doesn't define a separate file with the annotations
# Instead, the file name defines the identity and age
# E.g., 001a02.jpg (identity_age.jpg)
# Sometimes, the 'a' is lowercase, sometimes uppercase
# If multiple images of the same person at the same age are given, they are distinguished by a trailing letter,
# E.g., 010A07a.jpg and 010A07b.jpeg
print("Searching for files ...")
# Load paths to images and extract their annotations
types = ('/*.jpg', '/*.JPG', '/*.png', '/*.PNG')  # the tuple of file types
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(FGNet_root + files))
img_path = [os.path.basename(_) for _ in files_grabbed]
separated = [re.findall(r'\d+', _) for _ in img_path]

number, age = zip(np.array(separated).T)

number = [int(_) for _ in number[0]]
age = [int(_) for _ in age[0]]

# Create dataframe representing the dataset
df = pd.DataFrame({'img_path': img_path, 'id_num': number,
                  'age': age, 'database': ["FG-NET" for _ in img_path]})

# Prepend dataset path to file paths
df.img_path = FGNet_path + os.sep + df.img_path

# Assign identities to folders - each identity gets its own folder
df["folder"] = df.id_num


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
                color=colors[int(folder)%10], where='mid')

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
plt.savefig("FG-NET-splitting.png")

fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax, normalize=True)
plt.tight_layout()
plt.savefig("FG-NET-splitting-normalized.png")

# Dump the dataframe as a list of dict to JSON
f = open(output_file, "w+")
f.write(json.dumps(df.to_dict('records'), indent=4))

f.close()
print(f"FG-NET saved to {output_file}")
