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
import os

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

meaningful_concepts = {
    'airplane': ['airplane', 'outside_arm', 'engine', 'body', 'sand', 'light', 'blueness'],
    'automobile': ['car', 'door', 'manhole', 'sofa', 'chair', 'leather', 'house'],
    'bird': ['bird', 'fence', 'outside_arm', 'concrete', 'blackness', 'foot', 'person', 'leg', 'arm', 'stripes'], 
    'cat': ['cat', 'blackness', 'leg', 'stairs', 'neck', 'bottle', 'ear', 'foot', 'arm', 'curtain', 'stripes'],
    'deer': ['cow', 'fence', 'mountain', 'bench', 'foot', 'blueness', 'leg', 'arm', 'bus', 'redness', 'light'],
    'dog': ['dog', 'arm', 'muzzle', 'floor', 'polka_dots', 'bowl', 'grass', 'figurine', 'foot', 'leg'],
    'frog': ['beak', 'greenness', 'manhole', 'food', 'bed', 'foot', 'leg', 'arm', 'stripes'],
    'horse': ['horse', 'building', 'foot', 'leg', 'eye', 'water', 'fence', 'foot', 'arm', 'stripes', 'bannister'],
    'ship': ['water', 'flag', 'eye', 'hill', 'bridge'],
    'truck': ['bus', 'pipe', 'ashcan', 'polka_dots', 'clock', 'bumper']
}

def visualize_image_attr(
    original_image,
    attribution,
    method="heat_map",
    sign="absolute_value",
    plt_fig_axis=None,
    outlier_perc=2,
):
    if plt_fig_axis is None:
        _, axis = plt.subplots(1, 2, figsize=(12, 6))
    else:
        _, axis = plt_fig_axis

    # Visualize original image
    axis[0].imshow(np.transpose(original_image, (1, 2, 0)))
    axis[0].axis('off')
    axis[0].set_title('Original Image')

    # Process attribution for visualization
    attr = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))

    if sign == "absolute_value":
        attr = np.abs(attr)
    elif sign == "positive":
        attr = np.clip(attr, 0, 1)
    elif sign == "negative":
        attr = -np.clip(attr, -1, 0)

    if method == "heat_map":
        heatmap = np.sum(attr, axis=2)
        vmin, vmax = np.percentile(heatmap, [outlier_perc, 100 - outlier_perc])
        im = axis[1].imshow(heatmap, cmap='viridis', vmin=vmin, vmax=vmax)
        #plt.colorbar(im, ax=axis[1])
    else:
        raise NotImplementedError("Visualization method not implemented.")

    axis[1].axis('off')
    axis[1].set_title('Attribution Map')

    plt.tight_layout()
    return axis

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
    
class ModelWrapper_2(nn.Module):
    def __init__(self, classifier, clip_model, resolution):
        super(ModelWrapper_2, self).__init__()
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
        return x
    
# Function to load adversarial images
def load_adversarial_images(batch_file):
    # Load the tensor from the .pt file
    adversarial_images = torch.load(batch_file, map_location=device)
    return adversarial_images
    
    
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
classifier = torch.load("/data/gpfs/projects/punim2103/trained_pcbm_hybrid_cifar10_model__lam:0.0002__alpha:0.99__seed:42.ckpt", map_location=device)

batch_size = 128
resolution = 224  # specify the input resolution for your CLIP model

wrapped_model = ModelWrapper(classifier, model, resolution).to(device)
wrapped_model_2 = ModelWrapper_2(classifier, model, resolution).to(device)
wrapped_model.eval()
wrapped_model_2.eval()


test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(128))
test_loader = DataLoader(test_subset, batch_size=batch_size)

# Load the adversarial images
adversarial_dir = "/data/gpfs/projects/punim2103/results_clean/Linf/hpcbm/images/eps_0.001"
#adversarial_batches = [os.path.join(adversarial_dir, file) for file in os.listdir(adversarial_dir) if file.endswith('_adv.pt')]
adversarial_files = sorted(os.listdir(adversarial_dir))

save_dir = "/data/gpfs/projects/punim2103/ig_images/meaningful_concepts_adversarial_white_2"
ig = IntegratedGradients(wrapped_model_2)



# Analyze and print distances to important concepts for each image
for i, (images, labels) in enumerate(test_loader):
    # Get the distances for the batch

    images, labels = images.to(device), labels.to(device)
    #outputs, dist = wrapped_model(images)  # Shape [batch_size, num_concepts]
    
    
    # Load adversarial images for the current batch
    batch_filename = f"eps_0.001_batch_{i+1}_adv.pt"  # Construct the filename
    batch_file = os.path.join(adversarial_dir, batch_filename)
    adversarial_images = load_adversarial_images(batch_file).to(device)

    outputs, dist = wrapped_model(adversarial_images)

    # Convert the predicted and true labels to class names
    _, predicted = torch.max(outputs, 1)
    predicted_labels = [idx_to_class[label.item()] for label in predicted]
    true_labels = [idx_to_class[label.item()] for label in labels]


    # Make sure the shapes match, otherwise, there is a batch size mismatch
    
    assert adversarial_images.shape[0] == images.shape[0], "Batch sizes do not match."

    # Process each image in the batch
    for j in range(dist.size(0)):  # Iterate over the images in the batch
        # Get the distances for the j-th image
        distances = dist[j]

        # Get the 8 most influential concepts for the j-th image
        _, most_influential_indices = torch.topk(distances, 50)
        most_influential_concepts = [sorted_concept_list[idx] for idx in most_influential_indices]
        
        # Determine how many and which of the 8 concepts are in important_concepts for the true label
        relevant_concepts = set(meaningful_concepts[true_labels[j]])
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
        # Calculate and plot attributions for matched concepts

        for concept_name in matched_concepts:
            concept_idx = torch.tensor(sorted_concept_list.index(concept_name))
            concept_distance = distances[concept_idx].item()

            # Calculate the attribution for the current concept
            attributions = ig.attribute(adversarial_images[j].unsqueeze(0), baselines=1, target=int(concept_idx), return_convergence_delta=False, method='gausslegendre')
            
            # Convert the preprocessed image to numpy format for visualization
            np_img = adversarial_images[j].cpu().detach().numpy()
            
            # Visualize and save the attribution image
            _, axis = visualize_image_attr(
                original_image=np_img,
                attribution=attributions,
            )
            
            # Save the figure
            plt.savefig(f"{save_dir}/attribution_map_{i}_{j}_concept_{concept_name}_idx_{concept_idx.item()}_TrueLabel_{true_labels[j]}_PredLabel_{predicted_labels[j]}.png")
            
            # Clear the current figure to free memory before the next save
            plt.clf()

            print(f"Matched Important Concept: {concept_name}: {concept_distance:.3f}")

        
        print("\n---\n")
