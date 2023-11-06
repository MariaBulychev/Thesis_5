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






# Load CIFAR-10 test set
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(cifar10_testset, batch_size=2, shuffle=False)



for batch_idx, (images, _) in enumerate(test_loader):
    if batch_idx == 1:
        break

# Calculate the distances and logits
with torch.no_grad():
    distances = wrapped_model(images.to(device))

# Sort the distances and get indices of the smallest ones (top concepts)
sorted_indices = torch.argsort(distances, dim=1)

# Select the top 5 concept indices
top_concepts_indices = sorted_indices[:, :5]

save_dir = "/data/gpfs/projects/punim2103/ig_images/2"

for image_idx, image in enumerate(images):
    np_img = image.cpu().detach().numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(f"{save_dir}/original_image_{batch_idx}_{image_idx}.png")
    plt.close()

    # Process each concept
    for concept_idx in top_concepts_indices[image_idx]:
        # Here, we do not use the `attribute_image_features` function to avoid confusion.
        # We call `attribute` directly and pass the necessary arguments.
        # The `target` argument is the index in the logits that corresponds to the concept.

        # Also, since your wrapped_model.forward seems to return distances first, 
        # which are not the direct outputs towards which IntegratedGradients should
        # be computing attributions, you would need to make sure that your model
        # or a wrapper specifically returns the logits as outputs.
        # For now, let's assume you have a wrapper or a model variant that does this.

        attributions = ig.attribute(image.unsqueeze(0).to(device), target=int(concept_idx), return_convergence_delta=False)

        attr = attributions.squeeze().cpu().detach().numpy()
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(np_img, (1, 2, 0)))
        ax.imshow(np.transpose(attr, (1, 2, 0)), cmap='hot', alpha=0.5)
        ax.axis('off')
        plt.savefig(f"{save_dir}/attribution_image_{batch_idx}_{image_idx}_concept_{concept_idx.item()}.png")
        plt.close()