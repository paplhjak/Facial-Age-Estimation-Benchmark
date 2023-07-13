"""
Script which receives a database.json and computes the aligning bounding box
for each sample. The bounding box is either computed directly from provided landmarks
or is computed from landmarks detected by a facial detection model. If the sample does not
contain landmarks, but contains a bounding box annotation, the landmarks are detected only
in the specified bounding box.

The script currently requires:
1) alignment configuration file (! this is not the same as an experiment configuration file)
    - specifies path to data, as the database JSON contains only relative paths
    - specifies settings of the face detector
    - specifies settings of bounding box extension, before landmarks are detected in it
    - specifies the eye2eye and eye2mouth distance multipliers, which define the size of the aligned bounding box
    
2) JSON describing the dataset
    - contains samples to be aligned
"""

import yaml
import sys
import os
from lib.data_loaders import MyYamlLoader
from lib.utils import crop_image, get_alignment_transformation, pick_face_centered, pick_face_largest, bbox_area, extract_aligment_landmarks, draw_landmarks_and_bbox, draw_text
from prepare_data import normalize_img
import cv2
import json
import numpy as np
import torch
from typing import List, Tuple, Callable, Dict
from tqdm import tqdm

DEBUG = False


def get_model(config):
    """
    Extract RetinaFace from the InsightFace library.
    """

    import insightface
    from insightface.app import FaceAnalysis

    # initialize face detector
    app = FaceAnalysis(allowed_modules=['detection'],
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    app.prepare(ctx_id=0,
                det_size=config['detector']['det_size'],
                det_thresh=config['detector']['det_thresh'])

    # extract only the model of interest
    for taskname, model in app.models.items():
        if taskname == 'detection':
            return model


def retina_model_detect(img: np.ndarray, model, adaptive_shape=True) -> Tuple[List[int], List[int]]:
    """
    Detect faces using RetinaFace model. 
    Extract landmarks of interest and bounding boxes.

    Returns a list of discovered bounding boxes and a list of discovered landmarks. 
    The bounding box format is the same as for bbox. Landmarks are specified by a 
    dictionary with keys 'mouth_avg', 'eye_right', 'eye_left', where values are [col, row] coordinates.
    """

    if adaptive_shape:
        # landmarks detected with the input_size specified in the configuration file
        _, det_landmarks_default = model.detect(img)

        # remember input_size specified in the configuration
        original_input_size = model.input_size
        # compute nearest multiple of 32 pixels to the true image size
        det_shape = (np.maximum(
            np.array(img.shape[:2]) // 32, 1.) * 32).astype(int)
        # change detector settings
        model.input_size = det_shape

        # landmarks detected with input_size close to (nearest multiple of 32 pixels) the original shape
        _, det_landmarks = model.detect(img)

        # revert change of detector settings
        model.input_size = original_input_size

        # consider landmarks detected at both scales
        det_landmarks = np.concatenate(
            [det_landmarks, det_landmarks_default], axis=0)
    else:
        _, det_landmarks = model.detect(img)

    list_of_bboxes = []
    list_of_landmarks = []

    for landmarks in det_landmarks:

        # extract landmark coordinates
        mouth_right = landmarks[3]
        mouth_left = landmarks[4]
        mouth_avg = (mouth_left + mouth_right) / 2.
        eye_right = landmarks[0]
        eye_left = landmarks[1]

        # compute bounding box from the landmarks
        bbox = get_alignment_transformation(mouth_avg=mouth_avg.astype(int),
                                            eye_left=eye_left.astype(int),
                                            eye_right=eye_right.astype(int))

        list_of_bboxes.append(bbox)
        list_of_landmarks.append({'mouth_avg': mouth_avg.astype(int),
                                  'eye_left': eye_left.astype(int),
                                  'eye_right': eye_right.astype(int)})

    return list_of_bboxes, list_of_landmarks


def detect_landmarks_in_bbox(img: np.ndarray,
                             bbox: List[int],
                             crop_size: Tuple[int],
                             bbox_margin: Tuple[float],
                             detect: Callable[[np.ndarray], Tuple[List[int], List[int]]]) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    1) Crops the original image using bounding box specified in the benchmark.
    2) Detects faces in the cropped image and selects face closest to the center of the cropped image.
    3) Computes coordinates of the landmarks in the original image

    Args:
        img (np.ndarray): Shape [Height, Width, [B,G,R]] Image.
        bbox (List[int]): Bounding box [A_col, A_row, B_col, B_row, C_col, C_row, D_col, D_row] specified by the benchmark.
            The bbox is used to crop the image in step 1).
        crop_size (Tuple[int]): Size to which cropped bounding box is rescaled before detector call.
        bbox_margin (Tuple[float]): Margin used to scale up bounding box before crop.
        detect (Callable): Function, e.g., forward pass of a neural network, which returns a list of discovered bounding boxes
            and a list of discovered landmarks. The bounding box format is the same as for bbox. Landmarks are specified by a 
            dictionary with keys 'mouth_avg', 'eye_right', 'eye_left', where values are [col, row] coordinates.

    Returns:
        Tuple[Dict[str, np.ndarray], bool]: Landmarks, bool indicating whether alignment was successful
    """

    # if bbox is [0,0,0,...,0], return failure
    if all(v == 0 for v in bbox):
        {}, False

    # 1) crop the image using bounding box specified in the database
    cropped_img, M = crop_image(img,
                                bbox,
                                crop_size,
                                margin=bbox_margin,
                                one_based_bbox=True)

    # get inverse transform to the crop
    iM = cv2.invertAffineTransform(M)

    # 2) detect faces in the cropped image, i.e., landmark coordinates in the cropped image
    list_of_bboxes, list_of_landmarks = detect(
        cropped_img, adaptive_shape=False)

    # find out which bounding box is closest to center of the image
    centered_bbox, centered_landmarks, min_dist = pick_face_centered(
        list_of_bboxes, list_of_landmarks, crop_size)

    # no faces were detected
    if centered_landmarks is None:
        return {}, False

    # detected face is too small
    if bbox_area(centered_bbox) < (0.1 * crop_size[0] * crop_size[1]):
        return {}, False

    # 3) recompute coordinates of the discovered landmarks into basis of the original (not cropped) image
    mouth_avg = (
        iM@np.append(centered_landmarks['mouth_avg'], np.array(1.))).astype(int)
    eye_left = (
        iM@np.append(centered_landmarks['eye_left'], np.array(1.))).astype(int)
    eye_right = (
        iM@np.append(centered_landmarks['eye_right'], np.array(1.))).astype(int)

    return {'mouth_avg': mouth_avg.astype(int),
            'eye_left': eye_left.astype(int),
            'eye_right': eye_right.astype(int)}, True


def detect_landmarks_in_the_wild(img: np.ndarray,
                                 detect: Callable[[np.ndarray], Tuple[List[int], List[int]]],
                                 return_all=False) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Detects faces in the image and selects the largest face

    Args:
        img (np.ndarray): Shape [Height, Width, [B,G,R]] Image.
        detect (Callable): Function, e.g., forward pass of a neural network, which returns a list of discovered bounding boxes
            and a list of discovered landmarks. The bounding box format is the same as for bbox. Landmarks are specified by a 
            dictionary with keys 'mouth_avg', 'eye_right', 'eye_left', where values are [col, row] coordinates.
        return_all (bool): Flag. If True, returns a list of landmarks instead of a single face landmarks.

    Returns:
        Tuple[Dict[str, np.ndarray], bool]: Landmarks, bool indicating whether alignment was successful
    """

    # detect faces in the image
    list_of_bboxes, list_of_landmarks = detect(img, adaptive_shape=True)

    if return_all:
        return list_of_landmarks, len(list_of_landmarks) > 0

    # find out which bounding box is largest
    largest_bbox, largest_landmarks, min_dist = pick_face_largest(
        list_of_bboxes, list_of_landmarks)

    # no faces detected
    if largest_landmarks is None:
        return {}, False

    return largest_landmarks, True


if __name__ == '__main__':

    if len(sys.argv) != 3:
        sys.exit(
            f"usage: {sys.argv[0]} path/alignment_config.yaml path/database.json")

    # load config
    config_file = sys.argv[1]
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=MyYamlLoader)

    # load parameters for bounding box preprocessing
    bbox_crop_size = config['bbox']['crop_size']
    bbox_margin = config['bbox']['margin']

    model = get_model(config)

    # build callable function which detects faces
    def detect(img, adaptive_shape=False):
        return retina_model_detect(img, model, adaptive_shape=adaptive_shape)

    e2e = config['output_bbox']['eye_to_eye_scale_multipler']
    e2m = config['output_bbox']['eye_to_mouth_scale_multipler']

    # load the database
    db_path = sys.argv[2]
    with open(db_path, 'r') as stream:
        db = json.load(stream)

    count = 0
    for item_id, face in tqdm(enumerate(db), total=len(db)):

        img_file = config['data']['img_dir'] + face['img_path']

        if os.path.exists(img_file):

            if 'landmarks' in face:
                landmarks, success = extract_aligment_landmarks(
                    face['landmarks'])
                alignment_source = "landmarks"

            elif 'bbox' in face and len(face['bbox']) > 0:
                in_img = cv2.imread(img_file)
                landmarks, success = detect_landmarks_in_bbox(
                    in_img, face['bbox'], bbox_crop_size, bbox_margin, detect)
                alignment_source = "bbox"

            else:
                in_img = cv2.imread(img_file)
                landmarks, success = detect_landmarks_in_the_wild(
                    in_img, detect)
                alignment_source = "in the wild"

            if not success:
                face['aligned_bbox'] = []
                face['alignment_source'] = alignment_source
                continue

            # compute new bounding box, which aligns the face
            aligned_bbox = get_alignment_transformation(mouth_avg=landmarks['mouth_avg'],
                                                        eye_left=landmarks['eye_left'],
                                                        eye_right=landmarks['eye_right'],
                                                        eye_to_eye_scale_multipler=e2e,
                                                        eye_to_mouth_scale_multipler=e2m)

            aligned_bbox = [int(_) for _ in aligned_bbox]
            face['aligned_bbox'] = aligned_bbox
            face['alignment_source'] = alignment_source

            count += 1

            if DEBUG:
                in_img = cv2.imread(img_file)
                draw_landmarks_and_bbox(
                    in_img, bbox=aligned_bbox, landmarks=landmarks)
                draw_text(in_img, str(face['age']), aligned_bbox[2:4])
                cv2.imwrite(f'{item_id}.png', in_img)

        else:
            face['aligned_bbox'] = []
            face['alignment_source'] = "file not found"

    with open(f'{os.path.splitext(db_path)[0]}_aligned.json', 'w') as stream:
        json.dump(db, stream, indent=4)

    print(f"Aligned {count} faces out of {len(db)}")
