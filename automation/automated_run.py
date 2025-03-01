import subprocess

##################################
######## Train Models ############
##################################
# Define the base command template

base_command = "python train.py {config_path} {split} --wandb-disabled"

# List of configuration file paths
config_paths = [
    # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/other/Adience_256x256_resnet50_imagenet_dldl_v2.yaml",
    # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/Adience_256x256_resnet50_imagenet.yaml",
    # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/other/Adience_256x256_resnet50_imagenet_mean_variance.yaml",
    # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/other/Adience_256x256_resnet50_imagenet_coral.yaml",
    # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/other/Adience_256x256_resnet50_imagenet_soft_labels.yaml",
    # "facebase/configs/other/Adience_256x256_resnet50_imagenet_noisy_dldl_v2.yaml",
    "facebase/configs/other/Adience_256x256_resnet50_imagenet_dldl_v2.yaml",
]

# List of seeds to try
splits = [0, 1, 2, 3, 4]

# Loop through each combination of config path and seed
for config_path in config_paths:
    for split in splits:
        # Format the command
        command = base_command.format(config_path=config_path, split=split)
        print(f"Running command: {command}")
        
        # Execute the command and stream output to the console
        process = subprocess.run(command, shell=True)
        
        # Check if the command was successful
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}: {command}")
            break

##################################
######## Inject Noise ############
##################################
# Define the base command template

# base_command = "python inject_noise.py {data_path}/data_split{split}.csv {noise_matrix_file}"

# # List of configuration file paths
# data_paths = [
# # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet",
# "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_dldl_v2",
# # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_mean_variance",
# # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_soft_labels",
# # "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_coral"
# ]

# # List of seeds to try
# splits = [0, 1, 2, 3, 4]

# noise_matrix_file = "/media/vision/FastStorage-1/alireza-sm/Facial-Age-Estimation-Benchmark/noise_matrix_8_0.4_3.npy"

# # Loop through each combination of config path and seed
# for data_path in data_paths:
#     for split in splits:
#         # Format the command
#         command = base_command.format(data_path=data_path, split=split, noise_matrix_file=noise_matrix_file)
#         print(f"Running command: {command}")
        
#         # Execute the command and stream output to the console
#         process = subprocess.run(command, shell=True)
        
#         # Check if the command was successful
#         if process.returncode != 0:
#             print(f"Command failed with return code {process.returncode}: {command}")
#             break