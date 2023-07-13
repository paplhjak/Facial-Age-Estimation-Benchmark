"""
Implements a multitude of helper methods, mainly for facial alignment.
"""

import numpy as np
import os
from typing import Any, IO
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import cv2
import warnings


def create_dir(path):
    """
    Create directory if does not exists.

    Args:
        path (str): Path of directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def compute_error(metric,
                  true_label,
                  pred_label):
    """
    Computes a metric from predicted and true labels.
    """
    if metric == '0/1':
        err = np.mean(true_label != pred_label)
    elif metric == 'mae':
        err = np.mean(np.abs(true_label - pred_label))
    elif metric == 'cs5':
        err = np.mean(np.abs(true_label - pred_label) <= 5)
    else:
        print("Unknown metric!")
        err = []
    return err


def get_loss_matrix(n_y, loss):
    """
    Prepares a loss matrix for efficient computation of some metrics.
    The loss matrix holds the loss for every combination of predicted and true label.
    """
    L = torch.zeros(n_y, n_y)
    for y in range(n_y):
        for yy in range(n_y):
            if loss == '0/1':
                L[y, yy] = float(y != yy)
            elif loss == 'mae':
                L[y, yy] = torch.abs(torch.tensor(y-yy))
            else:
                raise Exception(f"The loss {loss} is not suported.")

    return L


def get_alignment_transformation(mouth_avg: np.ndarray,
                                 eye_left: np.ndarray,
                                 eye_right: np.ndarray,
                                 eyes_distance_only: bool = False,
                                 eye_to_eye_scale_multipler: float = 2.0,
                                 eye_to_mouth_scale_multipler: float = 1.8) -> np.ndarray:
    """
    Uses landmarks of the mouth and the eyes to construct a new bounding box, which does not need
    to be aligned with the axes and which can be used to normalize the size and orientation of images.

    Left / Right are meant from the observers perspective, i.e., left eye denotes the "biological right eye" etc.

    Args:
        mouth_avg (np.ndarray): [Col, Row] Coordinates of the cemter point of the mouth.
        eye_left (np.ndarray): [Col, Row] Coordinates of the left (biological) eye center.
        eye_right (np.ndarray): [Col, Row] Coordinates of the right (biological) eye center.
        eyes_distance_only (bool, optional): If True, the zoom is determined always by the distance of the eyes,
            and not the distance of the eyes to the mouth. If True, can result in undesirable "zoomed" in image for
            pictures of faces from the side. Defaults to False.
        eye_to_eye_scale_multipler (float, optional): Defines scaling of aligned image. Defaults to 2.0.
        eye_to_mouth_scale_multipler (float, optional): Defines scaling of aligned image. Defaults to 1.8.

    Returns:
        (np.ndarray): Array containing coordinates of a quadrilateral corners that forms the aligned image.
    """

    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_left - eye_right  # right eye to left eye vector
    eye_to_mouth = mouth_avg - eye_avg  # eye center to mouth center vector

    # left / right eye definitions are inconsistent w.r.t. the mouth position
    # swap the direction of eye to eye
    if angle_between(np.array([eye_to_mouth[1], -eye_to_mouth[0]]), eye_to_eye) > (np.pi / 2.):
        warnings.warn(
            "Left and Right eye (biological POV) positions are inconsistent w.r.t. the mouth position. Swapping them for computation of aligned bounding box.")
        eye_to_eye = -eye_to_eye

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]

    x /= (np.hypot(*x))  # normalize s.t. L2(x) = 1

    if eyes_distance_only:
        x *= np.hypot(*eye_to_eye) * eye_to_eye_scale_multipler
    else:
        x *= max(np.hypot(*eye_to_eye) * eye_to_eye_scale_multipler,
                 np.hypot(*eye_to_mouth) * eye_to_mouth_scale_multipler)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y,
                    c + x - y,
                    c + x + y,
                    c - x + y]).flatten().astype(int)

    return quad


def bbox_area(bbox):
    A = np.float32([bbox[0], bbox[1]])
    B = np.float32([bbox[2], bbox[3]])
    C = np.float32([bbox[4], bbox[5]])
    D = np.float32([bbox[6], bbox[7]])

    bbox_height = np.hypot(*(A-D))
    bbox_width = np.hypot(*(A-B))
    area = bbox_height*bbox_width
    return area


def pick_face_largest(list_of_bboxes,
                      list_of_landmarks):
    """
    Selects the largest bounding box within the image.

    Note that bounding box area outside of the image is included.
    """

    max_area = -np.inf
    picked_bbox = None
    picked_landmarks = None
    for bbox, landmarks in zip(list_of_bboxes, list_of_landmarks):

        area = bbox_area(bbox)
        if area > max_area:
            max_area = area
            picked_bbox = bbox
            picked_landmarks = landmarks

    return picked_bbox, picked_landmarks, max_area


def pick_face_centered(list_of_bboxes,
                       list_of_landmarks,
                       image_size):
    """
    Selects the bounding box closest to the center of the image.    
    """

    image_center = (np.array(image_size)/2.0).astype(int)
    min_distance = np.inf
    picked_bbox = None
    picked_landmarks = None
    for bbox, landmarks in zip(list_of_bboxes, list_of_landmarks):
        center = np.array([np.mean(bbox[0::2]), np.mean(bbox[1::2])])
        distance = np.hypot(*(center-image_center))

        if distance < min_distance:
            min_distance = distance
            picked_bbox = bbox
            picked_landmarks = landmarks

    return picked_bbox, picked_landmarks, min_distance


def crop_image(img: np.ndarray,
               bbox: List[int],
               out_size: Tuple[int],
               margin: Tuple[float] = (0, 0),
               one_based_bbox: bool = True):
    """
        Crop subimage around bounding box extended by a margin.

    Input:
     img 
     bbox = [A_col,A_row,B_col,B_row,C_col,C_row,D_col,D_row] bounding box
     out_size (cols,rows) size of output image
     margin 
     one_based_bbox [bool] if True assumes bbox to be given on 1-base coordinates
    Output:
     dst: output image [numpy array]
     M: affine transformation used for the crop

    Args:
        img (np.ndarray): Input image.
        bbox (List[int]): [A_col,A_row,B_col,B_row,C_col,C_row,D_col,D_row] bounding box, see README for more information.
        out_size (Tuple[int]): (cols,rows) size of output image.
        margin (Tuple[float], optional): (horizontal, vertical) margin; portion of bonding box size by which to extend the specified bounding box. Defaults to (0, 0).
        one_based_bbox (bool, optional): If True assumes that the bbox is given on coordinates starting with 1 instead of 0. Defaults to True.

    Returns:
        _type_: _description_
    """

    A = np.float32([bbox[0], bbox[1]])
    B = np.float32([bbox[2], bbox[3]])
    C = np.float32([bbox[4], bbox[5]])
    D = np.float32([bbox[6], bbox[7]])

    if one_based_bbox:
        A = A - 1
        B = B - 1
        C = C - 1
        D = D - 1

    ext_A = A + (A-B)*margin[0] + (A-D)*margin[1]
    ext_B = B + (B-A)*margin[0] + (B-C)*margin[1]
    ext_C = C + (C-D)*margin[0] + (C-B)*margin[1]

    pts1 = np.float32([ext_A, ext_B, ext_C])
    pts2 = np.float32([[0, 0], [out_size[0]-1, 0],
                      [out_size[0]-1, out_size[1]-1]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (out_size[0], out_size[1]))

    return dst, M


def extract_aligment_landmarks(landmarks: Dict[str, List[int]]) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Extracts facial landmarks used for face alignment.

    Args:
        landmarks (Dict[str, List[int]]):.

    Returns:
        Dict[str, np.ndarray]: Dictionary of landmarks. Landmarks are specified by keys 'mouth_avg', 'eye_right', 'eye_left', where values are [col, row] coordinates.
        bool: Indication whether extraction was successful.
    """
    eye_left = None
    eye_right = None
    mouth_avg = None

    # extract left eye center
    if 'left_eye_left_corner' in landmarks.keys() and \
            'left_eye_right_corner' in landmarks.keys():
        left_eye_left_corner = np.array(landmarks['left_eye_left_corner'])
        left_eye_right_corner = np.array(landmarks['left_eye_right_corner'])
        eye_left = np.mean(
            np.stack([left_eye_left_corner, left_eye_right_corner]), axis=0)
    elif 'left_eye_center' in landmarks.keys():
        eye_left = np.array(landmarks['left_eye_center'])

    # extract right eye center
    if 'right_eye_left_corner' in landmarks.keys() and \
            'right_eye_right_corner' in landmarks.keys():
        right_eye_left_corner = np.array(landmarks['right_eye_left_corner'])
        right_eye_right_corner = np.array(landmarks['right_eye_right_corner'])
        eye_right = np.mean(
            np.stack([right_eye_left_corner, right_eye_right_corner]), axis=0)
    elif 'right_eye_center' in landmarks.keys():
        eye_right = np.array(landmarks['right_eye_center'])

    # extract mouth center
    if 'mouth_left_corner' in landmarks.keys() and \
            'mouth_right_corner' in landmarks.keys():
        mouth_left_corner = np.array(landmarks['mouth_left_corner'])
        mouth_right_corner = np.array(landmarks['mouth_right_corner'])
        mouth_avg = np.mean(
            np.stack([mouth_left_corner, mouth_right_corner]), axis=0)
    elif 'mouth_center' in landmarks.keys():
        mouth_avg = np.array(landmarks['mouth_center'])

    # if extraction failed
    if eye_left is None or eye_right is None or mouth_avg is None:
        return {}, False
    else:
        return {'mouth_avg': mouth_avg.astype(int),
                'eye_left': eye_left.astype(int),
                'eye_right': eye_right.astype(int)}, True


def extract_L7_landmarks(landmarks: List[int]) -> Dict[str, np.ndarray]:
    """
    Extracts facial landmarks of interest from 7 facial landmarks defined below.

    (Biological)
    1. Left Eye Left Corner
    2. Left Eye Right Corner
    3. Right Eye Left Corner
    4. Right Eye Right Corner
    5. Nose Tip
    6. Mouth Left Corner
    7. Mouth Right Corner

    Args:
        landmarks (List[int]): List of 14 integers, which define 7 points of the dlib detector.

    Returns:
        Dict[str, np.ndarray]: Dictionary of landmarks. Landmarks are specified by keys 'mouth_avg', 'eye_right', 'eye_left', where values are [col, row] coordinates.
    """
    if len(landmarks) != 14 or all(v == 0 for v in landmarks):
        # Expected to get 14 numbers representing 7 points
        return {}, False

    left_eye_left_corner = np.array([landmarks[0], landmarks[1]])
    left_eye_right_corner = np.array([landmarks[2], landmarks[3]])
    right_eye_left_corner = np.array([landmarks[4], landmarks[5]])
    right_eye_right_corner = np.array([landmarks[6], landmarks[7]])
    mouth_left_corner = np.array([landmarks[10], landmarks[11]])
    mouth_right_corner = np.array([landmarks[12], landmarks[13]])

    eye_left = np.mean(
        np.stack([left_eye_left_corner, left_eye_right_corner]), axis=0)
    eye_right = np.mean(
        np.stack([right_eye_left_corner, right_eye_right_corner]), axis=0)
    mouth_avg = np.mean(
        np.stack([mouth_left_corner, mouth_right_corner]), axis=0)

    return {'mouth_avg': mouth_avg.astype(int),
            'eye_left': eye_left.astype(int),
            'eye_right': eye_right.astype(int)}, True


def extract_L21_landmarks(landmarks: List[int]) -> Dict[str, np.ndarray]:
    """
    Extracts facial landmarks of interest from 21 facial landmarks defined below.

    (Biological)
    1. Left Brow Left Corner
    2. Left Brow Center
    3. Left Brow Right Corner
    4. Right Brow Left Corner
    5. Right Brow Center
    6. Right Brow Right Corner
    7. Left Eye Left Corner
    8. Left Eye Right Corner
    9. Nose Root
    10. Right Eye Left Corner
    11. Right Eye Right Corner
    12. Left Ear
    13. Nose Left
    14. Nose Tip
    15. Nose Right
    16. Right Ear
    17. Mouth Left Corner
    18. Mouth Center Upper Lip
    19. Mouth Center Lower Lip
    20. Mouth Right Corner
    21. Chin Center

    Args:
        landmarks (List[int]): List of 42 integers, which define 21 points of the dlib detector.

    Returns:
        Dict[str, np.ndarray]: Dictionary of landmarks. Landmarks are specified by keys 'mouth_avg', 'eye_right', 'eye_left', where values are [col, row] coordinates.
    """
    if len(landmarks) != 42 or all(v == 0 for v in landmarks):
        # Expected to get 42 numbers representing 21 points
        return {}, False

    left_eye_left_corner = np.array([landmarks[12], landmarks[13]])
    left_eye_right_corner = np.array([landmarks[14], landmarks[15]])
    right_eye_left_corner = np.array([landmarks[18], landmarks[19]])
    right_eye_right_corner = np.array([landmarks[20], landmarks[21]])
    mouth_left_corner = np.array([landmarks[32], landmarks[33]])
    mouth_right_corner = np.array([landmarks[38], landmarks[39]])

    eye_left = np.mean(
        np.stack([left_eye_left_corner, left_eye_right_corner]), axis=0)
    eye_right = np.mean(
        np.stack([right_eye_left_corner, right_eye_right_corner]), axis=0)
    mouth_avg = np.mean(
        np.stack([mouth_left_corner, mouth_right_corner]), axis=0)

    return {'mouth_avg': mouth_avg.astype(int),
            'eye_left': eye_left.astype(int),
            'eye_right': eye_right.astype(int)}, True


def extract_L68_landmarks(landmarks: List[int]) -> Dict[str, np.ndarray]:
    """
    Extracts facial landmarks of interest from the dlib 68 landmarks.

    Args:
        landmarks (List[int]): List of 136 integers, which define 68 points of the dlib detector.

    Returns:
        Dict[str, np.ndarray]: Dictionary of landmarks. Landmarks are specified by keys 'mouth_avg', 'eye_right', 'eye_left', where values are [col, row] coordinates.
    """
    if len(landmarks) != 136 or all(v == 0 for v in landmarks):
        # Expected to get 136 numbers representing 68 points
        return {}, False

    left_eye_left_corner = np.array([landmarks[46*2-2], landmarks[46*2-1]])
    left_eye_right_corner = np.array([landmarks[43*2-2], landmarks[43*2-1]])
    right_eye_left_corner = np.array([landmarks[40*2-2], landmarks[40*2-1]])
    right_eye_right_corner = np.array([landmarks[37*2-2], landmarks[37*2-1]])
    mouth_left_corner = np.array([landmarks[55*2-2], landmarks[55*2-1]])
    mouth_right_corner = np.array([landmarks[49*2-2], landmarks[49*2-1]])

    eye_left = np.mean(
        np.stack([left_eye_left_corner, left_eye_right_corner]), axis=0)
    eye_right = np.mean(
        np.stack([right_eye_left_corner, right_eye_right_corner]), axis=0)
    mouth_avg = np.mean(
        np.stack([mouth_left_corner, mouth_right_corner]), axis=0)

    return {'mouth_avg': mouth_avg.astype(int),
            'eye_left': eye_left.astype(int),
            'eye_right': eye_right.astype(int)}, True


def draw_landmarks_and_bbox(img: np.ndarray,
                            bbox: List[int],
                            landmarks: Dict[str, np.ndarray],
                            line_color=(0, 0, 255),
                            line_thickness=2,
                            circle_thickness=2) -> None:
    """
    Draws a bounding box and eye and center of mouth landmarks into an image.

    Args:
        img (np.ndarray): Image loaded with OpenCV.
        bbox (List[int]): Bounding box [A_col, A_row, B_col, B_row, C_col, C_row, D_col, D_row].
        landmarks (Dict[str, np.ndarray]): Dictionary with keys 'mouth_avg', 'eye_right', 'eye_left' and values [col, row] coordinates.
        line_color (tuple, optional): Color of bounding box. Defaults to (0, 0, 255).
        line_thickness (int, optional): Thickness of bounding box. Defaults to 2.
        circle_color (tuple, optional): Color of landmarks. Defaults to (0, 255, 0).
        circle_thickness (int, optional): Thickness of landmarks. Defaults to 2.
    """

    if bbox is not None:
        cv2.line(img=img, pt1=bbox[0:2], pt2=bbox[2:4],
                 color=line_color, thickness=line_thickness)
        cv2.line(img=img, pt1=bbox[2:4], pt2=bbox[4:6],
                 color=line_color, thickness=line_thickness)
        cv2.line(img=img, pt1=bbox[4:6], pt2=bbox[6:8],
                 color=line_color, thickness=line_thickness)
        cv2.line(img=img, pt1=bbox[6:8], pt2=bbox[0:2],
                 color=line_color, thickness=line_thickness)

    if landmarks is not None:
        cv2.circle(img=img, center=landmarks['mouth_avg'],
                   radius=1, color=(255, 0, 0), thickness=circle_thickness)
        cv2.circle(img=img, center=landmarks['eye_right'],
                   radius=1, color=(0, 255, 0), thickness=circle_thickness)
        cv2.circle(img=img, center=landmarks['eye_left'],
                   radius=1, color=(0, 0, 255), thickness=circle_thickness)


def draw_text(img: np.ndarray,
              text: str,
              pos=(0, 0),
              font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)
