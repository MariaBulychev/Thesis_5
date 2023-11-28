#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-4:00:00
#SBATCH -p gpu-a100  
#SBATCH --gres=gpu:1        

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4

# Activate Environment 
source /data/gpfs/projects/punim2103/my_env/bin/activate

cd /data/gpfs/projects/punim2103/post_hoc_cbm

# Run your Python script with the specified parameters
echo "Running autoattack"

# Navigate to your directory, if needed

# Run the Python scripts
srun -n 1 python3 ft_both_different_lr_hpcbm.py --concept-bank=/data/gpfs/projects/punim2103/results/broden_clip:RN50_0.1_50.pkl --dataset="cifar10" --backbone-name="clip:RN50" --out-dir=/data/gpfs/projects/punim2103/joint_training --lam=2e-4 --last_num_ft=-1  --num-epochs=30 --clip-learning-rate=1e-7  