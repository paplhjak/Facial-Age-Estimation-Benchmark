# Face alignment

```
$ python prepare_alignment.py alignment_config.yaml input_database.json
```

The script creates a JSON database with aligned bounding boxes. The input is JSON database with faces defined by one of the following ways:

1. Using a set of landmarks. The landmarks are used to construct aligned bounding box.
2. Using a bounding box. In this case, the script finds landmarks and uses them to construct aligned bounding box.
3. No landmarks/no bounding box. In this case, the most dominant face and its landmarks are found by a face detector.

The output JSON is a copy of the input JSON augmented by "aligned_bbox" and "alignment_source" describing the way in which the face location was defined.

TBA: Describe how to define the aligned bbox give the landmarks.

The face aligment is configured in `alignment_config.yaml` which looks like:

```
data:
    img_dir: "facis/datasets/"

bbox:
    crop_size: [140, 140]
    margin: [0.25, 0.25]

output_bbox:
    eye_to_eye_scale_multipler: 1.92
    eye_to_mouth_scale_multipler: 1.89

detector:
    det_size: [160, 160]
    det_thresh: 0.5
```

# Work in progress ...

Explanation of the parameters ...
