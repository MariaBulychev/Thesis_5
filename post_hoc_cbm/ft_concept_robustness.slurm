#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-1:00:00
#SBATCH -p gpu-a100-short    
#SBATCH --gres=gpu:1        

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4

# Activate Environment 
source /data/gpfs/projects/punim2103/my_env/bin/activate

cd /data/gpfs/projects/punim2103/post_hoc_cbm

# Run your Python script with the specified parameters
echo "Concept robustness"

# Navigate to your directory, if needed

# Run the Python scripts
srun -n 1 python3 ft_concept_robustness.py