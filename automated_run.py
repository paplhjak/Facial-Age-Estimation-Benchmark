import subprocess

##################################
######## Train Models ############
##################################
# Define the base command template
base_command = "python train.py {config_path} {split} --wandb-disabled"

# List of configuration file paths
config_paths = [
    # "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/Adience_256x256_resnet50_imagenet.yaml",
    # "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/other/Adience_256x256_resnet50_imagenet_mean_variance.yaml",
    # "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/other/Adience_256x256_resnet50_imagenet_dldl_v2.yaml",
    # "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/other/Adience_256x256_resnet50_imagenet_soft_labels.yaml",
    "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/configs/other/Adience_256x256_resnet50_imagenet_coral.yaml",
]

# List of seeds to try
splits = [0, 1, 2]

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
# "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_4_3",
# "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_dldl_v2_4_3",
# "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_mean_variance_4_3",
# "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_soft_labels_4_3",
# "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/facebase/data/Adience_256x256_resnet50_imagenet_coral_4_3"
# ]

# # List of seeds to try
# splits = [0, 1, 2, 3, 4]

# noise_matrix_file = "/home/vision/alireza-sm/Facial-Age-Estimation-Benchmark/noise_matrix_8_0.4_3.npy"

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