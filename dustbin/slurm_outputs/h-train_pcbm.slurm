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
time srun -n 8 python3 train_pcbm_h.py --dataset="cifar10" --concept-bank=/data/gpfs/projects/punim2103/results/broden_clip:RN50_0.1_50.pkl --pcbm-path=/data/gpfs/projects/punim2103/train_results/pcbm_cifar10__clip:RN50__broden_clip:RN50_0__lam:0.0002__alpha:0.99__seed:42.ckpt --out-dir=/data/gpfs/projects/punim2103/h-train_results
