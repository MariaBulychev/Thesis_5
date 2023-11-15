import os
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import clip
import torch
import torch.nn as nn
from captum.attr import visualization as viz
import sys

sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')

concept_list = concept_list = [
    "air_conditioner", "basket", "blueness", "bumper", "ceramic", "countertop", "drinking_glass", 
    "fireplace", "greenness", "jar", "minibike", "pack", "plant", "water", "airplane", "bathroom_s", 
    "blurriness", "bus", "chain_wheel", "cow", "ear", "flag", "ground", "keyboard", "mirror", 
    "painted", "plate", "apron", "bathtub", "board", "bush", "chair", "cup", "earth", "floor", 
    "hair", "knob", "motorbike", "painting", "polka_dots", "arm", "beak", "body", "cabinet", 
    "chandelier", "curtain", "engine", "flower", "hand", "laminate", "mountain", "palm", "redness", 
    "armchair", "bed", "book", "can", "chest_of_drawers", "cushion", "exhaust_hood", "flowerpot", 
    "handle", "lamp", "mouse", "pane", "refrigerator", "ashcan", "bedclothes", "bookcase", 
    "candlestick", "chimney", "desk", "eye", "fluorescent", "handle_bar", "leather", "mouth", 
    "paper", "sand", "awning", "bedroom_s", "bottle", "canopy", "clock", "dining_room_s", "eyebrow", 
    "food", "head", "leg", "muzzle", "path", "snow", "back", "bench", "bowl", "cap", "coach", "dog", 
    "fabric", "foot", "headboard", "light", "neck", "paw", "sofa", "back_pillow", "bicycle", "box", 
    "car", "coffee_table", "door", "fan", "footboard", "headlight", "lid", "napkin", "pedestal", 
    "stairs", "bag", "bird", "brick", "cardboard", "column", "door_frame", "faucet", "frame", "hill", 
    "loudspeaker", "nose", "person", "street_s", "balcony", "blackness", "bridge", "carpet", 
    "computer", "doorframe", "fence", "glass", "horse", "manhole", "ottoman", "pillar", "stripes", 
    "bannister", "blind", "bucket", "cat", "concrete", "double_door", "field", "granite", "house", 
    "metal", "outside_arm", "pillow", "toilet", "base", "blotchy", "building", "ceiling", "counter", 
    "drawer", "figurine", "grass", "inside_arm", "microwave", "oven", "pipe", "tree"
]

idx_to_class = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

# Sort the list alphabetically
sorted_concept_list = sorted(concept_list)

# Function to load adversarial images
def load_adversarial_images(batch_file):
    # Load the tensor from the .pt file
    adversarial_images = torch.load(batch_file, map_location=device)
    return adversarial_images

# Load the model 
class ModelWrapper(nn.Module):
    def __init__(self, classifier, clip_model, resolution):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model
        
        # Define the preprocessing pipeline within the ModelWrapper
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        images = self.preprocess(images)
        print(f'images {images.shape}')
        features = self.clip_model.encode_image(images)
        print(f'features {features.shape}')
        out, x = self.classifier(features.float().to(device), return_dist = True)
        print(f'out {out.shape}, x {x.shape}')
        return out, x
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
classifier = torch.load("/data/gpfs/projects/punim2103/trained_pcbm_hybrid_cifar10_model__lam:0.0002__alpha:0.99__seed:42.ckpt", map_location=device)

batch_size = 128
resolution = 224  # specify the input resolution for your CLIP model

wrapped_model = ModelWrapper(classifier, model, resolution).to(device)
wrapped_model.eval()

# Load CIFAR-10 testset
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
# No longer creating a subset here, using the full test_dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Load the adversarial images
adversarial_dir = "/data/gpfs/projects/punim2103/results_clean/Linf/hpcbm/images/eps_1e-05"
#adversarial_batches = [os.path.join(adversarial_dir, file) for file in os.listdir(adversarial_dir) if file.endswith('_adv.pt')]
adversarial_files = sorted(os.listdir(adversarial_dir))

# Function to compare concepts
def compare_concepts(original_concepts, adversarial_concepts, top_n):
    return len(set(original_concepts[:top_n]).intersection(adversarial_concepts[:top_n])) / top_n * 100

# Results storage
# Initialize results list
results = []
batch = 0



# Iterate over both the original and adversarial images simultaneously
for i, (data, filename) in enumerate(zip(test_loader, adversarial_files)):
    print(batch)
    orig_images, labels = data
    orig_images, labels = orig_images.to(device), labels.to(device)
    #batch_file = os.path.join(adversarial_dir, filename)
    #adversarial_images = load_adversarial_images(batch_file).to(device)
    #print(adversarial_images.shape)

    # Load adversarial images for the current batch
    batch_filename = f"eps_1e-05_batch_{i+1}_adv.pt"  # Construct the filename
    batch_file = os.path.join(adversarial_dir, batch_filename)
    adversarial_images = load_adversarial_images(batch_file).to(device)

    # Make sure the shapes match, otherwise, there is a batch size mismatch
    assert adversarial_images.shape[0] == orig_images.shape[0], "Batch sizes do not match."

    # Get the model outputs for the original images
    orig_outputs, orig_dist = wrapped_model(orig_images)
    _, orig_predictions = torch.max(orig_outputs, 1)

    # Get the top 50 influential concepts for original images
    _, orig_influential_indices = torch.topk(orig_dist, 50, largest=False)
    orig_concepts = [[sorted_concept_list[idx] for idx in batch] for batch in orig_influential_indices]

    # Get the model outputs for the adversarial images
    adv_outputs, adv_dist = wrapped_model(adversarial_images)
    _, adv_predictions = torch.max(adv_outputs, 1)

    # Get the top 50 influential concepts for adversarial images
    _, adv_influential_indices = torch.topk(adv_dist, 50, largest=False)
    adv_concepts = [[sorted_concept_list[idx] for idx in batch] for batch in adv_influential_indices]

    break
    '''
    # Compare concepts and store results including correct and predicted labels
    for j in range(adversarial_images.size(0)):
        correct_label = idx_to_class[labels[j].item()]
        predicted_label_orig = idx_to_class[orig_predictions[j].item()]
        predicted_label_adv = idx_to_class[adv_predictions[j].item()]

        # Compute L2 and Linf norm between concept vectors
        l2_distance = torch.norm(orig_dist[j] - adv_dist[j], p=2)
        linf_distance = torch.norm(orig_dist[j] - adv_dist[j], p=float('inf'))

        # Initialize a dictionary to store the results for this image
        result = {
            'Image Index': i * batch_size + j,
            'Correct': correct_label,
            'Predicted Orig': predicted_label_orig,
            'Predicted Adv': predicted_label_adv,
            'L2 Distance': l2_distance.item(),
            'Linf Distance': linf_distance.item(),
            'Top-5': None,
            'Top-10': None,
            'Top-15': None,
            'Top-20': None,
            'Top-50': None
        }

        # Calculate the percentage of overlap for top N concepts
        for top_n in [5, 10, 15, 20, 50]:
            percent = compare_concepts(orig_concepts[j], adv_concepts[j], top_n)
            result[f'Top-{top_n}'] = percent

        results.append(result)

    batch += 1

# Convert results to a DataFrame and save as CSV
results_df = pd.DataFrame(results, columns=[
    'Image Index', 'Correct', 'Predicted Orig', 'Predicted Adv',
    'L2 Distance', 'Linf Distance', 'Top-5', 'Top-10', 'Top-15', 'Top-20', 'Top-50'
])
results_df.to_csv("/data/gpfs/projects/punim2103/results_clean/Linf/hpcbm/csv/concept_robustness_1e05.csv", index=False)


'''