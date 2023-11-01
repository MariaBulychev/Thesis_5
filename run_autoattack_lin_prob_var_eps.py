import os
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from autoattack import AutoAttack
import numpy as np


from torchvision.transforms import ToPILImage

class ModelWrapper(nn.Module):
    def __init__(self, classifier, clip_model, resolution):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model
        
        # Define the preprocessing pipeline within the ModelWrapper
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        images = self.preprocess(images)
        features = self.clip_model.encode_image(images)
        outputs = self.classifier(features.float().to(device))
        return outputs



# Load the model and preprocessing 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
batch_size = 16

print("load model")
classifier = torch.load("/data/gpfs/projects/punim2103/classifier_model_full.pth", map_location=device)
resolution = 224  # specify the input resolution for your CLIP model
wrapped_model = ModelWrapper(classifier, model, resolution).to(device)
wrapped_model.eval()

# load data
print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Define a list of epsilon values
epsilons = np.linspace(0.001, 0.1, 5)

batch = 0

for epsilon in epsilons:
    print("Epsilon:", epsilon)
    adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='custom',device= device, attacks_to_run=['apgd-ce'])
    #adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard', device=device)

    for images, labels in test_loader:
        print("start attack")
        batch += 1

        # Move images and labels to the appropriate device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)

        # Calculate accuracy on original images        
        correct = 0
        outputs = wrapped_model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        print('Accuracy on original images: {:.2f}%'.format(100 * correct / batch_size))

        results = adversary.run_standard_evaluation_individual(images, labels, bs=batch_size)
        #results = adversary.run_standard_evaluation(images, labels, bs=batch_size)
        #x_adv = results['apgd-ce']  # Get adversarial examples for the apgd-ce attack

        print("done")
        if batch == 10:
            batch = 0  # Reset batch counter for the next epsilon value
            break





