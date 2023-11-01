import os
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the model and preprocessing pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)


print("Load data")
#train_dataset = datasets.CIFAR10(root="./data", train=True, transform=preprocess, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=preprocess)

#train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)
#
classifier = torch.load("/data/gpfs/projects/punim2103/new_attempt_2_classifier_model_full.pth", map_location=device)
classifier.eval() # Set the model to evaluation mode


# Evaluate the classifier
print("evaluate")
classifier.eval()
correct = 0
total = 0

with torch.no_grad():
    batch = 0
    for images, labels in test_loader:
        print(batch)
        # images = preprocess(image).unsqueeze(0).to(device)
        images = images.to(device)
        features = model.encode_image(images)
        outputs = classifier(features.float().to(device))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        print(predicted.shape, labels.shape)
        correct += (predicted == labels.to(device)).sum().item()
        batch +=1
        #if batch == 2:
            #print("batch = 20")
            #break

print(f"Accuracy of the classifier on the test images: {100 * correct / total}%")
