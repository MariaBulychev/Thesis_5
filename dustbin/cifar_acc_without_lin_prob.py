import os
import clip
import torch
#from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from torch.nn.functional import cosine_similarity
from autoattack import AutoAttack



# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = './data'
#model, preprocess = clip.load('ViT-B/32', device)
model, preprocess = clip.load('RN50', device, download_root=out_dir)

# Download the dataset
#cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
dataset = datasets.CIFAR10(root="./data", download=True, train=False) #transform=preprocess, download=True)

correct_predictions = 0
total_images = 0

# Iterate over all images in the test set
for image, class_id in dataset:
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to(device)
    
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    
    # Calculate similarity and get the most similar label
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T)
    predicted_class = similarity.argmax(dim=-1).item()
    
    # Check if prediction is correct
    if predicted_class == class_id:
        correct_predictions += 1
    total_images += 1

# Calculate accuracy
accuracy = 100.0 * correct_predictions / total_images
print(f"Accuracy on CIFAR10 test set: {accuracy:.2f}%")



