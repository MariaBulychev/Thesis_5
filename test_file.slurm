#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-01:00:00
#SBATCH -p gpu-a100-short         
#SBATCH --gres=gpu:1  

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4

cd /data/gpfs/projects/punim2103/post_hoc_cbm 

# Activate Environment 
source /data/gpfs/projects/punim2103/my_env/bin/activate

cd /data/gpfs/projects/punim2103

# Run your Python script with the specified parameters
echo "Running test file"
time srun -n 8 python3 linear_probing_new_attempt.py
