import yaml
import sys
import os
from lib.data_loaders import MyYamlLoader
from lib.utils import crop_image, get_alignment_transformation, pick_face_centered, bbox_area
import cv2
import json
import numpy as np
from typing import List, Tuple, Callable, Dict
from tqdm import tqdm
import pickle
import pandas as pd
import argparse
import random
import copy


def apply_noise_to_list(data_list, npy_path, seed=4):
    """
    Applies label noise to a list-based dataset using a noise transition matrix.
    
    Parameters:
        data_list (list): The dataset in list format. Each element is a list with at least six elements,
                          where the last element (index 5) is a dictionary containing {'age': label}.
        npy_path (str): Path to the NumPy file containing the noise transition matrix.
        seed (int, optional): Seed for reproducibility.
    
    Returns:
        list: The modified dataset with noisy labels.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Load noise transition matrix
    noise_matrix = np.load(npy_path)
    
    # Identify all unique classes present in the dataset
    unique_classes = set(item[5] for item in data_list)
    
    # Create a copy of the dataset to modify
    noisy_data = copy.deepcopy(data_list)
    
    # Apply noise to each class
    for class_label in unique_classes:
        # Extract samples belonging to the current class
        class_samples = [item for item in noisy_data if item[5] == class_label]
        num_class_samples = len(class_samples)

        # Get the corresponding row from the noise matrix
        noise_row = noise_matrix[class_label]

        # Calculate the number of samples to convert to each class
        noisy_samples_count = (noise_row * num_class_samples).astype(int)

        # Keep track of already modified samples
        already_selected = set()

        # Apply noise based on the transition matrix
        for target_class, count in enumerate(noisy_samples_count):
            for _ in range(count):
                while True:
                    random_sample = random.choice(class_samples)
                    if id(random_sample) not in already_selected:
                        already_selected.add(id(random_sample))
                        break
                # Modify only the 'age' value inside labels
                random_sample[5] = target_class

    return noisy_data


def get_labels(face, tasks):
    labels = []
    for task in tasks:
        if "attribute" in task.keys():
            label = face[task['attribute']]
        else:
            label = face[task['tag']]

        if label in task['original_label']:
            labels.append(task['normalized_label'][task['original_label'].index(label)])
        else:
            return None

    return labels

def get_tasks(config):
    tasks = config['heads']
    for task in tasks:
        task['original_label'] = []
        task['normalized_label'] = []
        for c, label in enumerate(task['labels']):
            task['original_label'].extend(label)
            for i in range(len(label)):
                task['normalized_label'].append(c)
    return tasks

def normalize_img(img, bbox, input_size, input_extension, bbox_extension, return_affine_transform=False):

    out_size = (int(input_size[0]*(1+2*input_extension[0])),
                int(input_size[1]*(1+2*input_extension[1])))
    margin = (input_extension[0]+bbox_extension[0]+2*input_extension[0]*bbox_extension[0],
              input_extension[1]+bbox_extension[1]+2*input_extension[1]*bbox_extension[1])

    out_img, M = crop_image(img, bbox, out_size,
                            margin=margin, one_based_bbox=True)

    if return_affine_transform:
        return out_img, M

    return out_img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('config_file', type=str, help='Path to the config.yaml file')
    parser.add_argument('--inject_noise', type=str, required=False, help='Path to the noise transition matrix file')
    args = parser.parse_args()

    # load config
    config_file = args.config_file
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=MyYamlLoader)

    # load preprocessing parameters
    input_size = config['model']['input_size']
    input_extension = config['preprocess']['input_extension']
    bbox_extension = config['preprocess']['bbox_extension']

    # load benchmark (defines data and split of data to trn/val/tst folders)
    with open(config["data"]["benchmark"], 'r') as stream:
        benchmarks = yaml.load(stream, MyYamlLoader)

    # for each prediction task create a map between labels and normlaized label {0,1,...,num_labels-1}
    # normalized_label   original_label
    # 0                  'M'
    # 0                  'MALE'
    # 1                  'F'
    # 1                  'FEMALE'
    # ...
    tasks = config['heads']
    for task in tasks:
        task['original_label'] = []
        task['normalized_label'] = []
        for c, label in enumerate(task['labels']):
            task['original_label'].extend(label)
            for i in range(len(label)):
                task['normalized_label'].append(c)

    # create directory for preprocessed training images
    config_name = os.path.basename(config_file).split('.')[0]
    out_img_dir = config['data']['data_dir'] + f"{config_name}/images/"
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    # create face list and preprocessed training images ready for learning
    face_list = []
    count = -1
    for db_id, benchmark in tqdm(enumerate(benchmarks), total=len(benchmarks)):
        print("database:", benchmark['database'])

        folders = []
        for split in benchmark['split']:
            folders += split['trn'] + split['val'] + split['tst']
        folders = set(folders)

        db_path = os.path.dirname(
            config["data"]["benchmark"]) + '/' + benchmark['database']

        print(f"loading...")
        local_count = 0

        with open(db_path, 'r') as stream:
            db = json.load(stream)

        print("cropping images...")
        for item_id, face in tqdm(enumerate(db), total=len(db)):
            labels = get_labels(face, tasks)
            img_file = config['data']['img_dir'] + face['img_path']
            # bbox_exists = ("aligned_bbox" in face.keys() and len(face['aligned_bbox']) > 0) or ("bbox" in face.keys())
            if labels is not None and face['folder'] in folders and os.path.exists(img_file):
                count += 1
                local_count += 1

                # load face image
                in_img = cv2.imread(img_file)

                # if 'aligned_bbox' in face and len(face['aligned_bbox']) > 0:
                #     bbox = face['aligned_bbox']
                # else:
                #     bbox = face['bbox']

                # # normalize image
                # out_img = normalize_img(
                #     in_img, bbox, input_size, input_extension, bbox_extension)

                # save it to config['data_dir']/images/img_count.png
                out_img_path = out_img_dir + f"img{count:07d}.png"
                cv2.imwrite(out_img_path, in_img)

                # put it to face_list
                face_list.append([count, out_img_path, db_id,
                                 item_id, face['folder']] + labels)
            else:
                print('-------------------------------------------------')
                print('file exists:', os.path.exists(img_file))
                print('full annotation:', labels is not None)
                print('in folder:', face['folder'] in folders)
                # print('bbox exists:', bbox_exists)
                print(face)

        print(f"Accepted {local_count+1} faces out of {len(db)}")

    print(f"Total number of accepted faces {count+1}")
    
    if args.inject_noise:
        noisy_face_list = apply_noise_to_list(face_list, args.inject_noise)
        print(f"noisy_face_list created!")
    else:
        noisy_face_list = None

    # save face_list to face_list.csv
    face_list_file = config['data']['data_dir'] + \
        f'{config_name}/face_list.csv'
    f = open(face_list_file, "w+")
    for face in face_list:
        f.write(f"{face[0]}")
        for item in face[1:]:
            f.write(f",{item}")
        f.write("\n")
    f.close()

    #
    # folder[db_id][split][folder] -> {0..trn,1..val,2..tst}
    folder = []
    for benchmark in benchmarks:
        S = []
        for split_idx, split in enumerate(benchmark['split']):
            F = {}
            for part_idx, part in enumerate(['trn', 'val', 'tst']):
                for f in split[part]:
                    F[f] = part_idx
            S.append(F)
        folder.append(S)

    # create data_splitX.csv files
    for split_idx, split in enumerate(benchmark['split']):

        data_split_file = config['data']['data_dir'] + f'{config_name}/data_split{split_idx}.csv'
        
        with open(data_split_file, "w+") as f:
            for idx, face in enumerate(face_list):
                db_id = face[2]
                fol = folder[db_id][split_idx][face[4]]

                if noisy_face_list and fol in [0, 1]:
                    selected_face = noisy_face_list[idx]
                else:
                    selected_face = face

                f.write(f"{selected_face[0]},{selected_face[1]},{fol}")
                for item in selected_face[5:]:
                    f.write(f",{item}")
                for item in selected_face[5:]:
                    f.write(f",{item}")
                f.write(f",{config['training']['base_sigma']}")
                f.write("\n")
