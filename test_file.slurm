#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-01:00:00
#SBATCH -p gpu-a100-short         
#SBATCH --gres=gpu:1  

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4

cd /data/gpfs/projects/punim2103/post-hoc-cbm 

# Activate Environment 
source /data/gpfs/projects/punim2103/my_env/bin/activate

# Run your Python script with the specified parameters
echo "Running test file"
time srun -n 1 python3 linear_probing_resnet_eval.py
