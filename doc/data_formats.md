# Data Formats 
In the repository, we use a multitude of different file formats. We list the most commonly used files below.

For users only interested in reproducing our data splits, see definition of `database.json` and `benchmark.yaml`.

## face_list.csv
This CSV file can be used to link the normalized images (from `prepare_data.py`) with entries in the source JSON database. 

Each row of the file is a separate data sample, i.e., annotated face.

The file has the following columns:
- `face_id`: Unique IDs assigned to the normalized image.
- `img_path`: Path to the normalized image.
- `db_id`: ID of the database from which the image originates. For example, AgeDB might have db_id 0, AFAD might have db_id 1, etc.
- `item_id`: ID of the normalized image within its originating database. The ID is unique within the database, but does not have to be unique within the entire benchmark.
- `folder`: Integer used to separate the database into smaller "folders". The folders are then used to define splitting into parts for training, validation and testing.
- `labels`: Normalized labels, i.e., mapped to a set of integers (0,1,...,)

## data_split.csv

This CSV is similar to the face_list.csv, with a few exceptions. 

The file has the following columns:
- `face_id`: Unique IDs assigned to the normalized image.
- `img_path`: Path to the normalized image.
- `folder`: Integer used to separate the database into training (0), validation (1) and testing (2) parts. This is different from the `folder` definition in the `face_list.csv` 
- `labels`: Normalized labels, i.e., mapped to a set of integers (0,1,...,)

The data_split CSV file defines what data should be used for an experiment and how. 

However, for users only interested in replicating our data splits in their code, using the `data_split.csv` file may not me ideal. Because the `data_split.csv` file contains paths only to the normalized images, one might instead want to use the `database` JSON files instead.

## database.json

These files define a unified representation of all datasets.

Example of database entry:

```
{
        "img_path": "AFAD/AFAD-Full/15/111/638660-1.jpg",
        "id_num": 638660,
        "age": 15,
        "gender": "M",
        "database": "AFAD-Full",
        "folder": 0,
        "aligned_bbox": [
            -57,
            -95,
            372,
            -83,
            360,
            346,
            -69,
            334
        ],
        "alignment_source": "in the wild"
    }
```

To separate data into training, validation and testing parts, we use the `folder` attribute of the entry. The `folder` entry in a database JSON is identical to the `folder` attribute in `face_list.csv`, but different from the `folder` attribute in `data_split.csv`.

For datasets that define the training, validation and test splits (CLAP2016 and CACD2000), we use `folder` values in the database as follows.
- `0`: train
- `1`: val
- `2`: test

For the remaining datasets, we define a total of 10 folders (0-9) for each, see scripts in `facebase/to_json/`. 

The accompanying data splits can be found in `facebase/benchmarks`, however, for all of the datasets, 5 subject exlusive splits are defined by the folders as follows:

### Split 0
```
  - trn: [0,1,2,3,4,5]
    val: [6,7]
    tst: [8,9]
```
### Split 1
```
  - trn: [2,3,4,5,6,7]
    val: [8,9]
    tst: [0,1]
```
### Split 2
```
  - trn: [4,5,6,7,8,9]
    val: [0,1]
    tst: [2,3]
```
### Split 3
```
  - trn: [5,6,7,8,9,0]
    val: [1,2]
    tst: [3,4]
```
### Split 4
```
  - trn: [6,7,8,9,0,1]
    val: [2,3]
    tst: [4,5]
```

## benchmark.yaml

`Benchmark` files define a list of annotated databases and how to split them into training, validation and test parts.

The following example shows a `benchmark` which trains and evaluates on `AgeDB.yaml` and `CLAP2016.yaml`, at the same time. It defines two data splits for cross-validation (note that the CLAP2016 folders are identical in both splits, because CLAP2016 defines only one data split).

```
- database: path/AgeDB.yaml
  split:
    - trn: [0,1,2,3,4,5]
      val: [6,7]
      tst: [8,9]
    - trn: [2,3,4,5,6,7]
      val: [8,9]
      tst: [0,1]
- database path/CLAP2016.yaml
  split:
     - trn: [0]
       val: [1]
       tst: [2]
     - trn: [0]
       val: [1]
       tst: [2]
```

The specified `database` path in the benchmark is relative to where the `benchmark` file is stored.

## config.yaml

Each experiment is completely defined by a single `config.yaml` file. For a detailed explanation of the file structure, refer to ["Configuration"](configuration_file.md).


