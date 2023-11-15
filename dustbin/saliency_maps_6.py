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

# Sort the list alphabetically
sorted_concept_list = sorted(concept_list)

idx_to_class = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

important_concepts = {
    'airplane': ['airplane', 'fluorescent', 'outside_arm', 'engine', 'book', 'leather', 'foot', 'bag', 'knob', 'body'],
    'automobile': ['headlight', 'car', 'door', 'person', 'manhole', 'sofa', 'bookcase', 'back', 'palm', 'motorbike'],
    'bird': ['bird', 'fence', 'air_conditioner', 'loudspeaker', 'concrete', 'outside_arm', 'board', 'bicycle', 'eyebrow', 'lamp'],
    'cat': ['cat', 'blackness', 'leg', 'flowerpot', 'back_pillow', 'pack', 'stairs', 'neck', 'bottle', 'ear'],
    'deer': ['cow', 'paper', 'fan' 'fence', 'bumper', 'pillow', 'mountain', 'bench', 'foot', 'blueness'],
    'dog': ['dog', 'arm', 'muzzle', 'floor', 'pillow', 'headboard', 'polka_dots', 'bowl', 'grass', 'figurine'],
    'frog': ['beak', 'pipe', 'greenness', 'manhole', 'motorbike', 'food', 'chain_wheel', 'bed', 'headlight', 'handle'],
    'horse': ['horse', 'building', 'foot', 'leg', 'eye', 'water', 'bannister', 'balcony', 'fence', 'blotchy'],
    'ship': ['water', 'air_conditioner', 'flag', 'eye', 'street_s', 'napkin', 'bottle', 'bathtub', 'hill', 'bridge'],
    'truck': ['bus', 'mouse', 'apron', 'coach', 'pipe', 'ashcan', 'polka_dots', 'cardboard', 'clock', 'pack']
}

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
        features = self.clip_model.encode_image(images)
        out, x = self.classifier(features.float().to(device), return_dist = True)
        return out, x
    
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
classifier = torch.load("/data/gpfs/projects/punim2103/trained_pcbm_hybrid_cifar10_model__lam:0.0002__alpha:0.99__seed:42.ckpt", map_location=device)

batch_size = 2
resolution = 224  # specify the input resolution for your CLIP model

wrapped_model = ModelWrapper(classifier, model, resolution).to(device)
wrapped_model.eval()


test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(16))
test_loader = DataLoader(test_subset, batch_size=batch_size)






# Analyze and print distances to important concepts for each image
for i, (images, labels) in enumerate(test_loader):
    # Get the distances for the batch
    outputs, dist = wrapped_model(images)  # Shape [batch_size, num_concepts]
    
    # Convert the predicted and true labels to class names
    _, predicted = torch.max(outputs, 1)
    predicted_labels = [idx_to_class[label.item()] for label in predicted]
    true_labels = [idx_to_class[label.item()] for label in labels]

    # Process each image in the batch
    for j in range(dist.size(0)):  # Iterate over the images in the batch
        # Get the distances for the j-th image
        distances = dist[j]

        # Get the 8 most influential concepts for the j-th image
        _, most_influential_indices = torch.topk(distances, 50)
        most_influential_concepts = [sorted_concept_list[idx] for idx in most_influential_indices]
        
        # Determine how many and which of the 8 concepts are in important_concepts for the true label
        relevant_concepts = set(important_concepts[true_labels[j]])
        matched_concepts = set(most_influential_concepts) & relevant_concepts
        matched_concept_indices = [sorted_concept_list.index(concept) for concept in matched_concepts]
        
        # Print the true and predicted labels along with the most influential concepts
        print(f"Image {i * batch_size + j}:")
        print(f"True Class: {true_labels[j]}, Predicted Class: {predicted_labels[j]}")
        #print("Most Influential Concepts:")
        for concept_name in most_influential_concepts:
            concept_distance = distances[sorted_concept_list.index(concept_name)].item()
            #print(f"{concept_name}: {concept_distance:.3f}")
        
        # Print matched concepts from the important_concepts list
        print(f"Matched Important Concepts ({len(matched_concepts)}):")
        for concept_name in matched_concepts:
            concept_distance = distances[sorted_concept_list.index(concept_name)].item()
            print(f"{concept_name}: {concept_distance:.3f}")
        
        print("\n---\n")
