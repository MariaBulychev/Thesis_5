import torch
from captum.attr import IntegratedGradients
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import clip
import torch
import torch.nn as nn
from captum.attr import visualization as viz

concept_list = [
    "air_conditioner", "bannister", "bird", "bowl", "candlestick", "chandelier", "cow", "drawer",
    "fence", "footboard", "head", "lamp", "motorbike", "pack", "pillow", "street_s", "airplane",
    "base", "blackness", "box", "canopy", "chest_of_drawers", "cup", "drinking_glass", "field",
    "frame", "headboard", "leather", "mountain", "painted", "pipe", "stripes", "apron", "basket",
    "blind", "brick", "cap", "chimney", "curtain", "ear", "earth", "figurine", "fireplace", "granite",
    "grass", "greenness", "house", "light", "loudspeaker", "napkin", "palm", "plant", "toilet", "arm",
    "bathroom_s", "blotchy", "bridge", "car", "clock", "cushion", "engine", "exhaust_hood", "floor",
    "glass", "ground", "headlight", "leg", "lid", "mouth", "muzzle", "pane", "plate", "tree", "armchair",
    "bathtub", "blueness", "bucket", "cardboard", "coach", "desk", "dining_room_s", "eye", "eyebrow",
    "flower", "flowerpot", "hair", "jar", "keyboard", "knob", "laminate", "manhole", "metal", "microwave",
    "minibike", "mirror", "neck", "nose", "oven", "path", "paw", "person", "pillar", "refrigerator",
    "ashcan", "awning", "back", "back_pillow", "bag", "balcony", "bed", "bedclothes", "bedroom_s",
    "bench", "bicycle", "bottle", "book", "bookcase", "bush", "cabinet", "can", "carpet", "cat",
    "ceiling", "chain_wheel", "counter", "door", "door_frame", "doorframe", "double_door", "fan", "faucet",
    "food", "fluorescent", "foot", "handle", "handle_bar", "hill", "horse", "inside_arm", "manhole",
    "minibike", "outside_arm", "oven", "pedestal", "polka_dots", "redness", "refrigerator", "sand", "snow", "sofa"
]

# Sort the list alphabetically
sorted_concept_list = sorted(concept_list)





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
        features = self.clip_model.encode_image(images) # produces embeddings
        x = classifier.bottleneck.compute_dist(features.float()) # computes distances between embeddings and concepts. The output has shape torch.Size([2, 170])
        #probabilities = torch.softmax(x, dim=0)
        logits = classifier.bottleneck.classifier(x) # linear classifier maps x to the classes. The output has shape torch.Size([2, 10])
        #logits = self.classifier(features.float().to(device))
        return x    #     logits wenn man annimmt dass concepts die classes sind 



# Load the model and preprocessing 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
batch_size = 128

import sys
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')

print("load model")
classifier = torch.load("/data/gpfs/projects/punim2103/trained_pcbm_hybrid_cifar10_model__lam:0.0002__alpha:0.99__seed:42.ckpt", map_location=device)

resolution = 224  # specify the input resolution for your CLIP model
wrapped_model = ModelWrapper(classifier, model, resolution).to(device)
wrapped_model.eval()

# Integrated Gradients expects the model to return the logits directly,
# so you may need to modify your ModelWrapper to have an option to return the logits only.
# For now, let's assume it's already the case.

ig = IntegratedGradients(wrapped_model)


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
        plt.colorbar(im, ax=axis[1])
    else:
        raise NotImplementedError("Visualization method not implemented.")

    axis[1].axis('off')
    axis[1].set_title('Attribution Map')

    plt.tight_layout()
    return axis




# Load CIFAR-10 test set
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(cifar10_testset, batch_size=2, shuffle=False)

save_dir = "/data/gpfs/projects/punim2103/ig_images/3"

for batch_idx, (images, _) in enumerate(test_loader):
    if batch_idx == 1:
        break

    # Calculate the distances and logits
    with torch.no_grad():
        distances = wrapped_model(images.to(device))

    # Sort the distances and get indices of the smallest ones (top concepts)
    sorted_indices = torch.argsort(distances, dim=1)

    # Select the top 5 concept indices
    top_concepts_indices = sorted_indices[:, :3]

    

    # Define a black image baseline outside of the loop
    # Define a black image baseline outside of the loop
    black_image_baseline = 0

    for image_idx, image in enumerate(images):
        np_img = image.cpu().detach().numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.axis('off')
        plt.savefig(f"{save_dir}/original_image_{batch_idx}_{image_idx}.png")
        plt.close()

        for concept_idx in top_concepts_indices[image_idx]:
            attributions = ig.attribute(image.unsqueeze(0).to(device), baselines=black_image_baseline, target=int(concept_idx), return_convergence_delta=False, method='gausslegendre')

            visualize_image_attr(
            original_image=np_img,
            attribution=attributions,
            )
            
            plt.savefig(f"{save_dir}/attribution_map_{batch_idx}_{image_idx}_concept_{sorted_concept_list[concept_idx.item()]}_idx_{concept_idx}.png")
            plt.close()