from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm
from typing import Hashable
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

bad_annotations = ['part1/61_1_20170109142408075.jpg',
                   'part1/61_3_20170109150557335.jpg',
                   'part2/39_1_20170116174525125.jpg',
                   'part2/53__0_20170116184028385.jpg']

# Some files in UTKFace do not follow the [age]_[gender]_[race]_[date&time].jpg format
# They either omit some information and the leading '_' or omit the information only
# For this purpose, these files are manually corrected by the following dictionary
corrections = {"part1/61_1_20170109142408075.jpg": [61, 'F', 'Black', datetime.strptime('20170109142408', '%Y%m%d%H%M%S')],
               "part1/61_3_20170109150557335.jpg": [61, 'F', 'Indian', datetime.strptime('20170109150557', '%Y%m%d%H%M%S')],
               "part2/39_1_20170116174525125.jpg": [39, 'M', 'Black', datetime.strptime('20170116174525', '%Y%m%d%H%M%S')],
               "part2/53__0_20170116184028385.jpg": [53, 'F', 'White', datetime.strptime('20170116184028', '%Y%m%d%H%M%S')]}

# Define root to the dataset, the path to the output file and the number of desired folders
root = '/local/Faces'
UTKFace_path = 'UTKFace'
UTKFace_root = os.path.join(root, UTKFace_path)
output_file = '../benchmarks/databases/UTKFace.json'
nr_folders = 10

# The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
#   [age] is an integer from 0 to 116, indicating the age
#   [gender] is either 0 (male) or 1 (female)
#   [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
#   [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

print("Searching for files ...")
# Load paths to images and extract their annotations
types = ('/part*/*.jpg', '/part*/*.JPG', '/part*/*.png',
         '/part*/*.PNG')  # the tuple of file types
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(UTKFace_root + files))

img_path = []
age, gender, race, dateandtime = [], [], [], []

racedict = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
for file in files_grabbed:
    directory, basename = os.path.split(file)
    root, part = os.path.split(directory)

    path = os.path.join(part, basename)
    if path in bad_annotations:
        a, g, r, d = corrections[path]
        img_path.append(path)
        age.append(a)
        gender.append(g)
        race.append(r)
        dateandtime.append(d)
        continue

    a, g, r, d = basename.split('_')  # split file name into annotations
    d = d.split('.')[0]  # remove extension
    img_path.append(path)
    age.append(int(a))
    gender.append('M' if int(g) == 0 else 'F')
    race.append(racedict[int(r)])   
    dateandtime.append(datetime.strptime(d[:14], '%Y%m%d%H%M%S'))

dateandtime = [str(_) for _ in dateandtime]

# Create dataframe representing the dataset
df = pd.DataFrame({'img_path': img_path, 'age': age, 'race': race,
                  'acquired': dateandtime, 'database': ["UTKFace" for _ in img_path]})

# Prepend dataset path to file paths
df.img_path = UTKFace_path + os.sep + df.img_path

# Assign folders
folders = []
folder = 0
for img_path in tqdm(df.img_path):
    folders.append(folder)
    folder += 1
    folder %= nr_folders

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
                color=colors[int(folder) % 10], where='mid')

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
plt.savefig("UTKFace-splitting.png")

fig, axs = plt.subplots(ncols=len(attributes), figsize=(5*len(attributes), 4))
for attribute, ax in zip(sorted(attributes), axs):
    plot_folder_hist(df, attribute, ax, normalize=True)
plt.tight_layout()
plt.savefig("UTKFace-splitting-normalized.png")

# Dump the dataframe as a list of dict to JSON
f = open(output_file, "w+")
f.write(json.dumps(df.to_dict('records'), indent=4))

f.close()
print(f"UTKFace saved to {output_file}")
