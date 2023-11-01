#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-12:00:00
#SBATCH -p gpu-a100         
#SBATCH --gres=gpu:1        
#SBATCH --mail-user=maria.bulychev@student.unimelb.edu.au
#SBATCH --mail-type=ALL

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4

cd /data/gpfs/projects/punim2103/post-hoc-cbm 

# Activate Environment 
source /data/gpfs/projects/punim2103/my_env/bin/activate

# Run your Python script with the specified parameters
echo "Running learn_concepts_dataset.py"
time srun -n 8 python3 learn_concepts_dataset.py --dataset-name="broden" --backbone-name="clip:RN50" --C 0.001 0.01 0.1 1.0 10.0 --n-samples=50 --out-dir='/data/gpfs/projects/punim2103/results'
