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
        features = self.clip_model.encode_image(images) # embeddings
        x = classifier.bottleneck.compute_dist(features)
        logits = classifier.bottleneck.classifier(x)
        #logits = self.classifier(features.float().to(device))
        return x, logits



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

# Load CIFAR-10 test set
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
test_loader = DataLoader(cifar10_testset, batch_size=2, shuffle=False)

# Extract the first two images
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

x, logits = wrapped_model(images)
#logits = wrapped_model(images)
print(x)
print(x.shape)

print(logits)
print(logits.shape)
predicted_classes = torch.argmax(logits, dim=1)
print("True labels: "+str(labels))
print("Pred labels: "+str(predicted_classes))

'''
# Forward pass through the model to get the logits
logits = wrapped_model(images)
print(logits.shape)
# Get top-5 concepts for each class from the logits
# Assuming that wrapped_model has a bottleneck attribute with an analyze_classifier method
top_concepts = wrapped_model.classifier.bottleneck.analyze_classifier(k=5)
print(top_concepts)

weights = wrapped_model.classifier.residual_classifier.weight



print(weights)
print(weights.shape)
'''

'''
# Initialize Integrated Gradients
integrated_gradients = IntegratedGradients(wrapped_model)

# For each of the top concepts, compute attributions using Integrated Gradients
for i, concept in enumerate(top_concepts):
    # Get the index of the concept
    concept_index = concept.item()
    
    # Compute the attributions for the i-th concept
    attributions_ig = integrated_gradients.attribute(images, target=concept_index)
    
    # Process and visualize the attributions for each image
    for j in range(images.size(0)):
        # Process the attributions
        attribution = attributions_ig[j].cpu().detach().numpy()
        attribution = np.transpose(attribution, (1, 2, 0))
        
        # Visualize the attribution as a saliency map
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(attribution)
        plt.axis('off')
        plt.title(f'Image {j+1} - Concept {i+1}')
        
        plt.subplot(1, 2, 2)
        original_image = images[j].cpu().detach().numpy()
        original_image = np.transpose(original_image, (1, 2, 0))
        plt.imshow(original_image)
        plt.axis('off')
        plt.title(f'Original Image {j+1}')
        
        plt.show()
'''