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
    'airplane': ['airplane', 'fluorescent', 'outside_arm', 'engine', 'book'],
    'automobile': ['headlight', 'car', 'door', 'person', 'manhole'],
    'bird': ['bird', 'fence', 'air_conditioner', 'loudspeaker', 'concrete'],
    'cat': ['cat', 'blackness', 'leg', 'flowerpot', 'back_pillow'],
    'deer': ['cow', 'paper', 'fan' 'fence', 'bumper'],
    'dog': ['dog', 'arm', 'muzzle', 'floor', 'pillow'],
    'frog': ['beak', 'pipe', 'greenness', 'manhole', 'motorbike'],
    'horse': ['horse', 'building', 'foot', 'leg', 'eye'],
    'ship': ['water', 'air_conditioner', 'flag', 'eye', 'street_s'],
    'truck': ['bus', 'mouse', 'apron', 'coach', 'pipe']
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




# Loop over the testloader
for batch_idx, (images, true_labels) in enumerate(test_loader):
    # Move the images and labels to the same device as the model
    images, true_labels = images.to(device), true_labels.to(device)

    # Forward pass to get the output from the model
    with torch.no_grad():
        predictions, distances = wrapped_model(images)

    # Convert predictions to class indices
    _, predicted_indices = torch.max(predictions, 1)

    # Convert true labels and predicted indices to class labels
    true_class_labels = [idx_to_class[label.item()] for label in true_labels]
    predicted_class_labels = [idx_to_class[label.item()] for label in predicted_indices.cpu().numpy()]

    # Analyze and print distances to important concepts for each image
    for i, (true_label, predicted_label) in enumerate(zip(true_class_labels, predicted_class_labels)):
        print(f"Batch {batch_idx}, Image {i}:")
        print(f"True Class: {true_label}, Predicted Class: {predicted_label}")

        # Get the distances for the current image
        image_distances = distances[i]

        # Print min and max distances for the image
        min_dist = torch.min(image_distances).item()
        max_dist = torch.max(image_distances).item()
        print(f"Min Distance: {min_dist:.3f}, Max Distance: {max_dist:.3f}")
        
        # Retrieve the relevant concepts based on the predicted class
        relevant_concepts = important_concepts[predicted_label]

        # Print the distances to the important concepts
        print("Distances to important concepts:")
        for concept in relevant_concepts:
            concept_index = sorted_concept_list.index(concept)
            concept_distance = image_distances[concept_index].item()
            print(f"{concept}: {concept_distance:.3f}")
        print("\n---\n")

    