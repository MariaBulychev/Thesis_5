import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import clip
import os
import sys
import sys
import clip
import torch
import torch.nn as nn
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoattack import AutoAttack
from torchvision.transforms import ToPILImage

# Assuming ModelWrapper is defined above, as in your provided code
import sys
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')


def load_adversarial_images(folder_path, dataset, batch_size):
    # Load adversarial images and use labels from the first 1000 images of the dataset
    adv_data_list = []
    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith('_adv.pt') and idx < len(dataset):
            batch_path = os.path.join(folder_path, filename)
            adv_data = torch.load(batch_path, map_location=device)
            adv_data_list.append(adv_data)

    # Only use the first 1000 labels
    labels = [label for _, label in dataset][:1000]

    # Assuming that adversarial images are split into batches
    # You might need to concatenate them here if necessary
    adv_data_tensor = torch.cat(adv_data_list, dim=0)
    adv_labels_tensor = torch.tensor(labels)

    # Create a TensorDataset and DataLoader
    adv_dataset = torch.utils.data.TensorDataset(adv_data_tensor, adv_labels_tensor)
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=batch_size, shuffle=False)
    return adv_loader

# Evaluate the model on adversarial images
def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

    accuracy = total_correct / total_images * 100
    return accuracy

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
        logits = self.classifier(features.float().to(device))
        return logits

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load('RN50', device)
classifier = torch.load("/data/gpfs/projects/punim2103/trained_pcbm.ckpt", map_location=device)
classifier.eval()

resolution = 224  # specify the input resolution for your CLIP model
wrapped_model = ModelWrapper(classifier, clip_model, resolution).to(device)
wrapped_model.eval()

# Load the CIFAR10 dataset to get labels
cifar10_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
cifar10_subset = Subset(cifar10_dataset, range(1000))

# Load adversarial images
adv_images_folder = '/data/gpfs/projects/punim2103/autoattack_results/l_2/hybrid_pcbm'
batch_size = 128  # or the batch size that you used
adv_data_loader = load_adversarial_images(adv_images_folder, cifar10_subset, batch_size)

# Evaluate the model
accuracy = evaluate(wrapped_model, adv_data_loader, device)
print(f'Accuracy on adversarial images: {accuracy:.2f}%')
