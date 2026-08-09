#!/bin/bash

# Get the current date and time
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Set the relevant directories
code_directory="INSERT/PATH/TO/PHIMO-MRM/CODE/DIRECTORY"
anaconda_directory="INSERT/PATH/TO/ANACONDA/DIRECTORY"

# Set the output filename with the timestamp
output_filename="$code_directory/iml-dl/results/recon_t2star/logs/log_${timestamp}.txt"

# activate conda env:
source $anaconda_directory/bin/activate
conda activate cupa_T2_star

# change directory
cd $code_directory/iml-dl/

# Run the Python script and redirect the output to the timestamped file
nohup python -u ./core/Main.py --config_path ./projects/recon_t2star/configs/config_train.yaml > "$output_filename" &

