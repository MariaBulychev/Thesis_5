#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-1:00:00
#SBATCH -p gpu-a100-short         
#SBATCH --gres=gpu:1        
#SBATCH --mail-user=maria.bulychev@student.unimelb.edu.au
#SBATCH --mail-type=ALL

# Load required modules
module purge
module load foss/2022a
module load Python/3.10.4

cd /data/gpfs/projects/punim2103/post_hoc_cbm

# Activate Environment 
source /data/gpfs/projects/punim2103/my_env/bin/activate

# Run your Python script with the specified parameters
echo "Running learn_concepts_dataset.py"


time srun -n 1 python3 train_pcbm_h_finetuning.py --concept-bank=/data/gpfs/projects/punim2103/edited_concepts/results/broden_clip:RN50_0.1_50.pkl --checkpoint-path=/data/gpfs/projects/punim2103/joint_training/finetuning_both_40_epochs_different_lr/final_model_finetuned_-1_cliplayers_1e-07_clip_lr_1e-07_pcbm_lr.pth --out-dir=/data/gpfs/projects/punim2103/joint_training/finetuning_both_40_epochs_different_lr/h_pcbm --dataset="cifar10" --l2-penalty=0.01 --backbone-name="clip:RN50"
