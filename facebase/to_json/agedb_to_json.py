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
AgeDB_path = 'AgeDB/AgeDB'
AgeDB_root = os.path.join(root, AgeDB_path)
output_file = '../benchmarks/databases/AgeDB.json'
nr_folders = 10

# AgeDB doesn't define a separate file with the annotations
# Instead, the file name defines the identity, age and the gender
# E.g., 0_MariaCallas_35_f.jpg (number_name_age_gender.jpg)
print("Searching for files ...")
# Load paths to images and extract their annotations
img_path = [os.path.basename(_) for _ in glob.glob(AgeDB_root + '/*.jpg')]
separated = [_.split('_') for _ in img_path]
number, name, age, gender = zip(np.array(separated).T)


number = [_ for _ in number[0]]
name = [_ for _ in name[0]]
age = [int(_) for _ in age[0]]
gender = ['M' if 'm' in g else 'F' for g in gender[0]]

# Create dataframe representing the dataset
df = pd.DataFrame({'img_path': img_path, 'name': name, 'age': age,
                  'gender': gender, 'number': number, 'database': ["AgeDB" for _ in img_path]})

# Prepend dataset path to file paths
df.img_path = AgeDB_path + os.sep + df.img_path

# Assign identities to folders
unique_ids = df.name.unique()
np.random.shuffle(unique_ids)
ages = [df.age[df.name == name] for name in unique_ids]
id2folder = defaultdict(int)
folder2ages = defaultdict(lambda: np.array([]))

print("Splitting the files into subject exclusive folders with balanced distributions of age ...")
for unique_id, id_ages in tqdm(zip(unique_ids, ages), total=len(unique_ids)):
    most_represented_age = Counter(id_ages).most_common()[0][0]
    counts = [sum(folder2ages[folder] == most_represented_age)
              for folder in range(nr_folders)]

    folder_with_least_counts = np.argmin(counts)
    id2folder[unique_id] = int(folder_with_least_counts)
    folder2ages[folder_with_least_counts] = np.append(
        folder2ages[folder_with_least_counts], id_ages)

for name in unique_ids:
    df.loc[df.name == name, "folder"] = id2folder[name]

df.folder = df.folder.astype(int)

# Assert that the splitting is subject exclusive
print("Making sure the splitting is subject exclusive ...")
for id_num in unique_ids:
    assert len(df.folder[df.name == name].unique()) == 1


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
plt.savefig("AgeDB-Full-splitting.png")

fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax, normalize=True)
plt.tight_layout()
plt.savefig("AgeDB-Full-splitting-normalized.png")

# Dump the dataframe as a list of dict to JSON
f = open(output_file, "w+")
f.write(json.dumps(df.to_dict('records'), indent=4))

f.close()
print(f"AgeDB saved to {output_file}")
