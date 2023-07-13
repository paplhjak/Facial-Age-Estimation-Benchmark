# Configuration file

Each experiment is completely defined by a single `config.yaml` file. An example of such a file is:

```
data:
    benchmark: "facebase/benchmarks/MORPH.yaml"
    img_dir: "/home/user/Data/age_estimation_data/"
    data_dir: "prepared_data/"
    output_dir: "training_results/"

heads:
    - tag: "age"
     attribute: "age"
     label: [[0],[1],...,[90,91,92,...,100]]
     weight: 1
     metric: ['mae','cs5']
     visual_metric: ['mae]

model:
    architecture: "resnet18"
    use_pretrained: False
    input_size: [128,128]

optimizer:
    num_workers: 4
    num_epochs: 100
    batch_size: 100
    improve_patience: 5
    lr: 0.001
    betas: [0.9,0.999]
    eps: 0.00000001
    algo: "adam"
    use_amp: False

preprocess:
    bbox_extension: [0.25,0.25]
    input_extension: [0.05,0.05]
    trn: {path: "lib/augmentation_configs/128x128/jitter+hflip.json" }
    val: {path: "lib/augmentation_configs/128x128/center_crop.json"}
```

The configuration file is separated into sections `data`, `heads`, `model`, `optimizer` and `preprocess`. These sections define:

- `data`: 
    - What data to use (`benchmark`).
    - Where the raw data is stored (`img_dir`).
    - Where to save processed data, ready for training (`data_dir`).
    - Where to store experiment results (`output_dir`).
- `heads`: 
    - What prediction task/tasks the model will be solving, see ["Prediction Task Definition"](prediction_task_definition.md).
- `model`: 
    - Architecture of the model.
    - Input size of the model. This also specifies the size of the clean data.
- `optimizer`: 
    - Optimization protocol for training.
- `preprocess`: 
    - Strategy for converting raw data into clean data.
    - Data pipelines for the model input data, used during training and deployment.

- [Model](backbone_support.md)
- [Optimizer](optimizer.md)
- [Preprocess](preprocess_augment.md)