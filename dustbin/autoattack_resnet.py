import torch
import clip
from torchvision import datasets, transforms
from torch.nn.functional import cosine_similarity
from autoattack import AutoAttack

# Load the CLIP model and the CIFAR-10 dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("RN50", device=device)
dataset = datasets.CIFAR10(root="./data", transform=transform, download=True)

# Prepare text prompts
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in dataset.classes]).to(device)

# Extract text features which remain constant
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# Wrapper around the CLIP model to return logits instead of image features
class CLIPWrapper(torch.nn.Module):
    def __init__(self, clip_model, text_features):
        super(CLIPWrapper, self).__init__()
        self.clip_model = clip_model
        self.text_features = text_features

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        # Calculate the similarity (logit) between image features and text features
        logits = cosine_similarity(image_features, self.text_features)
        return logits

clip_model_for_aa = CLIPWrapper(model, text_features).to(device)

# Prepare the test set
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate using AutoAttack
adversary = AutoAttack(clip_model_for_aa, norm='Linf', eps=0.3, version='standard')

correct = 0
total = 0

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    adv_images = adversary.run_standard_evaluation(images, labels, bs=images.size(0))
    # For simplicity, just check if the perturbed images are correctly classified
    preds = torch.argmax(clip_model_for_aa(adv_images), dim=1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)

robust_accuracy = correct / total
print(f"Robust accuracy against AutoAttack: {robust_accuracy:.2%}")
