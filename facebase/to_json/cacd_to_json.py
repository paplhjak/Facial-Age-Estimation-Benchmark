from collections import defaultdict, Counter
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from tqdm import tqdm
from typing import Hashable
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Define root to the dataset, the path to the output file and the number of desired folders
root = '/local/Faces'
CACD_path = 'CACD2000/CACD2000'
CACD_root = os.path.join(root, CACD_path)
annotations_path = os.path.join(root, 'CACD2000', 'celebrity2000_meta.mat')
output_file = '../benchmarks/databases/CACD2000.json'

# Annotations are defined in a matlab file
# The code the load the matlab file is 'not pretty' in python
print("Reading annotations file ...")
mat = loadmat(annotations_path)
mdata = mat['celebrityImageData']
mdata_ = mat['celebrityData']

# Extract the annotations
id2name = {identity_[0]: name_[0][0] for name_, identity_ in zip(
    mdata_['name'][0][0], mdata_['identity'][0][0])}

mdtype = mdata.dtype
ndata = {n: mdata[n][0][0] for n in mdtype.names}

# Prepend dataset path to file paths
img_path = [os.path.join(CACD_path, _[0][0]) for _ in (ndata['name'])]

identity = (ndata['identity']).flatten()
age = ndata['age'].flatten()
birth = ndata['birth'].flatten()
lfw = ndata['lfw'].flatten()
rank = ndata['rank'].flatten()
year = ndata['year'].flatten()

name = []
for identity_ in identity:
    name.append(id2name[identity_])

# Build dataframe from extracted fields
df = pd.DataFrame({'name': name, 'img_path': img_path, 'id_num': identity,
                  'age': age, 'birth': birth, 'lfw': lfw, 'rank': rank, 'year': year, 'database': ["CACD" for _ in img_path]})


df["folder"] = [-1 for name in df.name]

df.loc[df["rank"]<=5, "folder"] = 2
df.loc[df["rank"]<=2, "folder"] = 1
df.loc[df["rank"]>5, "folder"] = 0

unique_ids = df.id_num.unique()
# Assert that the splitting is subject exclusive
print("Making sure the splitting is subject exclusive ...")
for id_num in unique_ids:
    assert len(df.folder[df.id_num == id_num].unique()) == 1


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
plt.savefig("CACD-splitting.png")

fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax, normalize=True)
plt.tight_layout()
plt.savefig("CACD-splitting-normalized.png")

# Dump the dataframe as a list of dict to JSON
f = open(output_file, "w+")
f.write(json.dumps(df.to_dict('records'), indent=4))

f.close()
print(f"CACD saved to {output_file}")
