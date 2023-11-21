import torch
import clip
#from models import PosthocLinearCBM  # Import the PCB model class
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the path to the saved checkpoint file
checkpoint_path = '/data/gpfs/projects/punim2103/joint_training/final_model_finetuned_-1_cliplayers_1e-07_clip_lr_1e-07_pcbm_lr_acc 8362.pth'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load the CLIP model state_dict
clip_model_state_dict = checkpoint['clip_model_state_dict']

# Load the PCBM model state_dict
pcbm_model_state_dict = checkpoint['pcbm_model_state_dict']

# Load the CLIP model architecture
clip_model, preprocess = clip.load('RN50', device, jit=False)

# Set the CLIP model state_dict
clip_model.load_state_dict(clip_model_state_dict)

# Instantiate the PCB model (assuming it's a Posthoc Concept Bank Model)
#pcb_model = PosthocLinearCBM(input_dim=clip_model.output_dim, num_classes=YOUR_NUM_CLASSES)  # Replace YOUR_NUM_CLASSES
pcbm = torch.load('/data/gpfs/projects/punim2103/trained_pcbm.ckpt', map_location = device)

# Set the PCB model state_dict
pcbm.load_state_dict(pcbm_model_state_dict)

# Save the CLIP model and PCB model as separate full models
clip_model_path = '/data/gpfs/projects/punim2103/joint_training/finetuning_full_models/clip_model.pth'
torch.save(clip_model, clip_model_path)

pcb_model_path = '/data/gpfs/projects/punim2103/joint_training/finetuning_full_models/pcbm.pth'
torch.save(pcbm, pcb_model_path)

# Now you have saved the full CLIP model and PCB model as separate files.
