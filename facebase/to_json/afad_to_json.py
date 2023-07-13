from collections import defaultdict, Counter
from tqdm import tqdm
from typing import Hashable
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Define root to the dataset, the path to the output file and the number of desired folders
root = '/local/Faces'
AFAD_path = 'AFAD/AFAD-Full'
AFAD_root = os.path.join(root, AFAD_path)
output_file = '../benchmarks/databases/AFAD-Full.json'
nr_folders = 10

# AFAD doesn't define a separate file with the annotations
# Instead, the data is organized in folders that define the age and the gender
# E.g., age/gender/XXXXX-X.jpg
# For 'M' (male) the corresponding folder is '/111/', for 'F' (female) the folder is '112'
print("Searching for files ...")
# Load paths to images and extract their annotations
samples = [os.path.normpath(_).split(os.path.sep)[-3:]
           for _ in glob.glob(AFAD_root + '/*/*/*.jpg')]
print(samples)
age, gender, img_path = zip(np.array(samples).T)

id_num = [int(_.split('-')[0]) for _ in img_path[0]]
img_path = [os.path.join(AFAD_path, a, g, p)
            for p, a, g in zip(img_path[0], age[0], gender[0])]
age = [int(_) for _ in age[0]]
gender = ['M' if g == '111' else 'F' for g in gender[0]]

# Create dataframe representing the dataset
df = pd.DataFrame({'img_path': img_path, 'id_num': id_num, 'age': age,
                  'gender': gender, 'database': ["AFAD-Full" for _ in img_path]})

unique_ids = df.id_num.unique()
# Assign identities to folders
id2folder = defaultdict(int)
print("Splitting the files into subject exclusive folders ...")
folder = 0
for unique_id in tqdm(unique_ids):
    id2folder[unique_id] = folder
    folder += 1
    folder %= nr_folders


# Add the folder annotation to the dataframe
folders = []
for id_num in tqdm(df.id_num):
    folders.append(id2folder[id_num])

df["folder"] = folders


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
plt.savefig("AFAD-Full-splitting.png")

fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax, normalize=True)
plt.tight_layout()
plt.savefig("AFAD-Full-splitting-normalized.png")

# Dump the dataframe as a list of dict to JSON
f = open(output_file, "w+")
f.write(json.dumps(df.to_dict('records'), indent=4))

f.close()
print(f"AFAD saved to {output_file}")
