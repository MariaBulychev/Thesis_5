#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-2:00:00
#SBATCH -p gpu-a100-short  
#SBATCH --gres=gpu:1    
#SBATCH --mem=32G

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4

# Activate Environment 
source /data/gpfs/projects/punim2103/my_env/bin/activate

cd /data/gpfs/projects/punim2103/ZSRobust

# Run your Python script with the specified parameters
echo "Finetuning"

# Navigate to your directory, if needed

# Run the Python scripts
srun -n 1 python3 finetuning.py --batch_size 128  --root /data/gpfs/projects/punim2103/data --dataset cifar10 --name maria --train_eps 1 --train_numsteps 2 --train_stepsize 1 --device='cpu'