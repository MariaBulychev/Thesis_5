#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-6:00:00
#SBATCH -p gpu-a100   
#SBATCH --gres=gpu:1        

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4

# Activate Environment 
source /data/gpfs/projects/punim2103/my_env/bin/activate

cd /data/gpfs/projects/punim2103

# Run your Python script with the specified parameters
echo "Running autoattack"

# Navigate to your directory, if needed

# Run the Python scripts
srun -n 1 python3 skip_2.py 0.00001 
