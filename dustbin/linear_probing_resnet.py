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
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=preprocess, download=True)
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=preprocess)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Extract features from train dataset
train_features = []
train_labels = []

with torch.no_grad():
    #batch = 0 
    for image, label in train_loader:
        #images = preprocess(image).unsqueeze(0).to(device)
        #print("batch")
        images = image.to(device)
        features = model.encode_image(images)
        train_features.append(features)
        train_labels.append(label)
        #batch +=1
        #if batch == 20:
            #print("batch = 20")
            #break

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
torch.save(classifier, "/data/gpfs/projects/punim2103/classifier_model_full.pth")


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
        images = image.to(device)
        features = model.encode_image(images)
        outputs = classifier(features.float().to(device))
        print("outputs shape: "+str(outputs.shape))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        print(predicted.shape, labels.shape)
        correct += (predicted == labels.to(device)).sum().item()
        batch +=1
        #if batch == 2:
            #print("batch = 20")
            #break

print(f"Accuracy of the classifier on the test images: {100 * correct / total}%")
