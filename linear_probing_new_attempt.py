import os
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the model without its original preprocessing pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load('RN50', device)

# Define your custom preprocessing pipeline
resolution = 224  # Assuming the resolution for RN50 model
preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

print("Load data")
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)
# Extract features from train dataset
train_features = []
train_labels = []

preprocess_fn = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

with torch.no_grad():
    #batch = 0 
    for image, label in train_loader:
        image = preprocess_fn(image)

        images = image.to(device)
        features = model.encode_image(images)
        train_features.append(features)
        train_labels.append(label)

train_features = torch.cat(train_features)
train_labels = torch.cat(train_labels)

# Define a simple linear classifier
print("Define classifier")
classifier = nn.Linear(train_features.size(1), 10).to(device)  # 10 classes for CIFAR10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)





# Train the linear classifier
num_epochs = 10
for epoch in range(num_epochs):
    print("start epoch")
    classifier.train()
    running_loss = 0.0
    for i, (features, labels) in enumerate(zip(train_features.split(128), train_labels.split(128))):
        optimizer.zero_grad()
        outputs = classifier(features.float().to(device))#.unsqueeze(0))
        loss = criterion(outputs, labels.to(device))#.unsqueeze(0))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader)}")

#save 
print("save classifier")
#torch.save(classifier.state_dict(), "/data/gpfs/projects/punim2103/classifier_model.pth")
torch.save(classifier, "/data/gpfs/projects/punim2103/new_attempt_2_classifier_model_full.pth")


