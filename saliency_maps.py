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





# Load the model and preprocessing 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
batch_size = 128

import sys
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')

print("load model")
classifier = torch.load("/data/gpfs/projects/punim2103/trained_pcbm_hybrid_cifar10_model__lam:0.0002__alpha:0.99__seed:42.ckpt", map_location=device)


ig = IntegratedGradients(model)


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




# Load CIFAR-10 test set
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(cifar10_testset, batch_size=2, shuffle=False)

save_dir = "/data/gpfs/projects/punim2103/ig_images/7"
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
cat_index = sorted_concept_list.index('cat')
blackness_index = sorted_concept_list.index('blackness')
leg_index = sorted_concept_list.index('leg')
flowerpot_index = sorted_concept_list.index('flowerpot')
back_pillow_index = sorted_concept_list.index('back_pillow')
#print(cat_index) = sorted_concept_list.index('cat')


for batch_idx, (images, labels) in enumerate(test_loader):
    if batch_idx == 1:
        break


    # Calculate the distances and logits
    with torch.no_grad():
        distances = model(images.to(device), return_dist = True)
        print("distances "+str(distances))

        # To get the top 6 maximum values and indices for each row (image):
        _, top6_indices = torch.topk(distances, 6, dim=1)

        for i in range(distances.shape[0]):  # Loop over each image
            print(f"Image {i}:")
            print(f"  True label: {cifar10_classes[labels[i]]}")  # Printing the true label of the image
            print("Distance to cocept cat: "+str(distances[0][cat_index]))
            print("Distance to cocept blackness: "+str(distances[0][blackness_index]))
            print("Distance to cocept leg: "+str(distances[0][leg_index]))
            print("Distance to cocept flowerpot: "+str(distances[0][flowerpot_index]))
            print("Distance to cocept back_pillow: "+str(distances[0][back_pillow_index]))
            for j in range(6):  # Loop to print top 6 concepts
                concept_index = top6_indices[i][j].item()
                concept_name = sorted_concept_list[concept_index]  # Retrieve the concept name using the index
                print(f"  Concept {j+1}: {concept_name}")
                
    '''
    # Sort the distances and get indices of the smallest ones (top concepts)
    #sorted_indices = torch.argsort(distances, dim=1)

    # Sort the distances to get indices of the largest ones (top concepts)
    sorted_indices = torch.argsort(distances, dim=1, descending=True)

    # Select the top 5 concept indices
    top_concepts_indices = sorted_indices[:, :3]

    predicted_labels = predictive_model(images)

    

    # Define a black image baseline outside of the loop
    # Define a black image baseline outside of the loop
    black_image_baseline = 0

    

    for image_idx, image in enumerate(images):
        print(f"Image {batch_idx}_{image_idx}")
        top_concepts = top_concepts_indices[image_idx]
        for rank, concept_idx in enumerate(top_concepts):
            concept = sorted_concept_list[concept_idx.item()]
            weight = distances[image_idx, concept_idx].item()
            print(f"  Concept rank {rank + 1}: {concept} with weight {weight}")

        np_img = image.cpu().detach().numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.axis('off')
        #plt.savefig(f"{save_dir}/original_image_{batch_idx}_{image_idx}.png")
        plt.close()

        for concept_idx in top_concepts_indices[image_idx]:
            attributions = ig.attribute(image.unsqueeze(0).to(device), baselines=black_image_baseline, target=int(concept_idx), return_convergence_delta=False, method='gausslegendre')

            visualize_image_attr(
            original_image=np_img,
            attribution=attributions,
            sign = "positive"
            )
            
            print("Saving image "+str(batch_idx)+"_"+str(image_idx)+" Concept "+str(sorted_concept_list[concept_idx.item()]) +" idx "+str(concept_idx)  )
            plt.savefig(f"{save_dir}/attribution_map_{batch_idx}_{image_idx}_concept_{sorted_concept_list[concept_idx.item()]}_idx_{concept_idx}_TrueLabel_{labels[image_idx]}_PredLabel_{predicted_labels[image_idx]}.png")
            plt.close()
'''
