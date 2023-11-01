import os
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from autoattack import AutoAttack


class ModelWrapper(nn.Module):
    # Model = classifier 
    # clip_model = Resnet50
    def __init__(self, classifier, clip_model):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model

    def forward(self, images):
        features = self.clip_model.encode_image(images)
        outputs = self.classifier(features.float().to(device))
        return outputs

# Load the model and preprocessing 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
batch_size = 16

print("load model")
classifier = torch.load("/data/gpfs/projects/punim2103/classifier_model_full.pth", map_location=device)
wrapped_model = ModelWrapper(classifier, model).to(device)
wrapped_model.eval()

# load data
print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# define the attacker
#epsilon = 8./255.
#adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='custom',device= device, attacks_to_run=['apgd-ce'])

epsilon = 0.001
adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='custom',device= device, attacks_to_run=['apgd-ce'])


correct = 0
for images, labels in test_loader:
    print("start attack")
    images, labels = images.to(device), labels.to(device)

    # Calculate accuracy on original images
    outputs = wrapped_model(images)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()

    
    results = adversary.run_standard_evaluation_individual(images, labels, bs=batch_size)
    
    x_adv = results['apgd-ce']  # Get adversarial examples for the apgd-ce attack
    
    # Combine the original and adversarial images for visualization
    combined_images = torch.cat((images, x_adv), dim=0)
    
    # Save the combined images (original + adversarial) in a grid format
    grid_path = "/data/gpfs/projects/punim2103/results_images/1/combined_grid.png"
    save_image(combined_images, grid_path, nrow=batch_size//4)  # Assuming you want a 4x4 grid if you have 16 images
    
    # Save the original images as tensors
    tensor_path_original = "/data/gpfs/projects/punim2103/results_images/1/original_tensors.pt"
    torch.save(images.cpu(), tensor_path_original)
    
    # Save the adversarial images as tensors
    tensor_path_adv = "/data/gpfs/projects/punim2103/results_images/1/adversarial_tensors.pt"
    torch.save(x_adv.cpu(), tensor_path_adv)

    print("done")
    break

# Print the accuracy
print('Accuracy on original images: {:.2f}%'.format(100 * correct / batch_size))

