import clip
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoattack import AutoAttack




#from torchvision.transforms import ToPILImage

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



# Load the model and preprocessing 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
batch_size = 128

import sys
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')

print("load model")
#classifier = torch.load("/data/gpfs/projects/punim2103/classifier_model_full.pth", map_location=device)

classifier = torch.load("/data/gpfs/projects/punim2103/trained_pcbm_hybrid_cifar10_model__lam:0.0002__alpha:0.99__seed:42.ckpt", map_location=device)

resolution = 224  # specify the input resolution for your CLIP model
wrapped_model = ModelWrapper(classifier, model, resolution).to(device)
wrapped_model.eval()

# load data
print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(1000))
test_loader = DataLoader(test_subset, batch_size=batch_size)


# define the attacker
epsilon = 0.001
#adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='custom',device= device, attacks_to_run=['apgd-ce'])
adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard',device= device)

import torch


batch = 0
# Prepare to save results to CSV
csv_path = '/data/gpfs/projects/punim2103/autoattack_results/hybrid_pcbm_2/results.csv'

with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Epsilon", "Initial Accuracy", "Robust Accuracy", "Max Perturbation"])  # Header

    for images, labels in test_loader:
        print("start attack")
        batch += 1
        print("batch "+str(batch))
        
        # Move images and labels to the appropriate device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)

        outputs = wrapped_model(images)
        _, predicted = torch.max(outputs, 1)
        initial_acc = (predicted == labels).sum().item()
        print('Initial Accuracy for Batch {}: {:.2f}%'.format(batch, 100 * initial_acc / batch_size))

        x_adv, robust_accuracy, res = adversary.run_standard_evaluation(images, labels, bs=batch_size)

        # Save adversarial examples (x_adv) for this batch as a tensor
        torch.save(x_adv, f'/data/gpfs/projects/punim2103/autoattack_results/hybrid_pcbm_2/eps_{epsilon}_batch_{batch}_adv.pt')

        # Save results to CSV
        csv_writer.writerow([epsilon, 100 * initial_acc / batch_size, robust_accuracy, res.item()])

        print("done")
        if batch * batch_size >= 1000:  # Stop after processing 1000 images
            break





