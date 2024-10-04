<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/paplhjak/Facial-Age-Estimation-Benchmark">
    <img src="logo_a.png" alt="Logo" width="300" height="300">
  </a>
  <h1 align="center">Using the Repository</h1>
</div>

# Installation

To run this project on your local PC, you need to install the necessary packages.

First, create a conda environment called `age_conda_env` by running the following command in your terminal:

```
conda env create -f environment.yaml
```

Once the environment has been created, activate it by running:

```
conda activate age_conda_env
```

To run the scripts on a GPU, you will need to reinstall the `torch`, `torchvision`, and `torchaudio` packages. You can do this by running the following command:

```
$CONDA_PREFIX/bin/pip install torch torchvision torchaudio
```

Once you have installed the packages and activated the `age_conda_env` environment, you can run the project by executing the relevant scripts.

In case you decide to reproduce the results, you may find the resnet50 model pretrained on IMDB-Clean at the following [link](https://paplham.cloud/index.php/s/TmtPyz4ABkYkL7E).

# Pipeline

In this document, we describe the pipeline of obtaining and preprocessing the data, defining the experiment configuration file and training the models.

## Step 0: Preparing databases
### Download the datasets
First, we need to obtain annotated data. In other words, we need to download datasets such as AgeDB, AFAD or MORPH. 

Next, we need to create a unified representation for all of the datasets. This can be achieved by the following steps. Alternatively, rest of **step 0** can be skipped and the prepared representations (*JSON databases*) can be obtained from the["Databases repository"](https://github.com/paplhjak/Facial-Age-Estimation-Benchmark-Databases).  

### Create database JSONs (Optional)
Some datasets come as a collection of images where the annotation is included in the image name. Other datasets provide a separate file which provides the annotations.
To work with multiple datasets efficiently, we hence need to create a unified representation. To this end, we use the scripts in `facebase/to_json/`. Output of each such script is a JSON file to which we refer to as a **Database**. 

Example of a database entry:

```
{
        "img_path": "AFAD/AFAD-Full/15/111/638660-1.jpg",
        "id_num": 638660,
        "age": 15,
        "gender": "M",
        "database": "AFAD-Full",
        "folder": 0
    }
```

The database JSON file contains a list of data samples, each described by an image path, some attributes (e.g., age and gender) and a *folder*. The folder describes whether the sample is used for training, validation or testing. For more information, refer to ["Using the data splits"](using_the_data_splits.md).

### Facial alignment (Optional)
For tasks such as age estimation, we do not want to use the entirety of the image as input to our model. Instead, we want to extract a part of the image corresponding to the face of the person. To this end, we need to locate the face and possibly also normalize its position (**alignment**).

For this, we utilize the script `prepare_alignment.py` to detect faces and to create aligned bounding boxes. The script takes the JSON database and adds a new entry `aligned_bbox` to each of the samples. Afterwards, the images can be further processed and the faces cropped out.

For more information on the facial alignment, see ["Alignment"](prepare_alignment.md).

We also provide the resulting JSON files in a separate ["Databases repository"](https://github.com/paplhjak/Facial-Age-Estimation-Benchmark-Databases), so that our exact data splits and alignment setup can be reproduced by others.

## Step 1: Preparing benchmarks (Optional)
After having prepared the database JSON files describing the datasets, we need to define which datasets (or equivalently, which databases) to use and how. We want to use some samples for training, some for validation and some for testing. We might also want to use multiple datasets, not only one. To this end, we construct YAML files which define what datasets are used and how they are used. We refer to these YAML files as **Benchmarks** and we define which samples are used for training, validation or testing by their corresponding `folder` attribute. For more information, refer to ["Using the data splits"](using_the_data_splits.md).

We provide the benchmark files we use in `facebase/benchmarks/`.

An example of a benchmark YAML is:

```
- database: facebase/benchmarks/databases/MORPH_aligned.json
  tag: "MORPH"
  split:
     - trn: [0,1,2,3,4,5]
       val: [6,7]
       tst: [8,9]
     - trn: [2,3,4,5,6,7]
       val: [8,9]
       tst: [0,1]
```

It defines two data splits (for 2-fold cross-validation) and separates the data samples from the `MORPH_aligned.json` database by the `folder` attribute into training, validation and testing part for both data splits.

## Step 2: Defining a configuration file (Optional)

Next, we need to define the experiment configuration file. Each experiment is completely defined by a single `config.yaml` file. It specifies what data to use (by specifying a benchmark file), how to preprocess the data, how to costruct the model and how to train it.

We provide a basic selection of configuration files we use in `facebase/configs/`.

The configuration files are described in detail in the ["Configuration"](doc/configuration_file.md).

## Step 3: Preparing data for experiments

Now we are ready to prepare the data for our experiments by running the script `prepare_data.py` as

```
$ python prepare_data.py config.yaml
```

Outputs of the script are stored in a newly created directory, specified by the configuration file in `config['data']['data_dir']`.

The script prepares the following:
- face_list.csv file which assigns a unique ID to each image in the benchmark
- data_split.csv files which define what images are to be used for training, which are to be used for validation and which are to be used for testing
- It then creates a subdirectory `images` and begins cropping out the faces as specified by the `aligned_bbox` attribute. The process is described in more detail in ["Preprocessing"](preprocess_augment.md).

The data to prepare is specified by the selected benchmark in the configuration file, see ["Benchmarks"](benchmarks.md).

After the cropped faces are extracted, the model can be trained.

## Step 4: Training

After the data for an experiment are prepared, the model can be trained by calling:

```
$ python train.py config.yaml split
```

The script trains and evaluates the neural network on a prediction problems defined by data_splitX.csv file where X is the split number. The script must be issued for each split separaterly, e.g.:

```
$ python train.py config.yaml 0
$ python train.py config.yaml 1
...
$ python train.py config.yaml 4
```

After the script is done executing, the output is stored in `config['data']['output_dir']/config/splitX/` directory. It contains:

1. "model_cpu.pt" and "model_gpu.pt", which are JIT scripted versions of the model.
2. "evaluation.pt" which contains predictions of the model on all data samples. The file can then be processed by `evaluate.py` to obtain the model performance on different datasets.
3. the checkpoint files.

## Step 5: Evaluation

Lastly, the model performance can be evaluated. Predictions of the model on all data samples are saved at the end of the training. Hence, the predictions only need to be processed by calling:

```
$ python evaluate.py config.yaml
```

This script takes predictions for each data split and evaluates the metrics defined for each head. 

For more info how to define the metrics see the ["Prediction task definition"](prediction_task_definition.md).

The output of the script is a HTML report stored in `config['data']['output_dir']/config/evaluation_index.html`.

## Step 6: Deployment & Utility scripts

These scripts were removed from the current version.
