import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

# Assuming these are your custom modules
from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections
import clip
from torchvision import datasets, transforms

# Define the configuration for the model
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num-epochs", default=10, type=int)
    return parser.parse_args()

# Function to train for one epoch
def train_epoch(model, clip_model, train_loader, criterion, clip_optimizer, pcbm_optimizer, preprocess, device):
    model.train()
    clip_model.train()  # Set CLIP model to training mode

    total_loss = 0
    for inputs, labels in train_loader:
        inputs = preprocess(inputs).to(device)
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass through CLIP model
        features = clip_model.module.encode_image(inputs)

        if torch.isnan(features).any():
            print("NaN detected in model outputs")
            continue  # Skip this batch or handle NaNs as needed

        # Forward pass through PCBM
        outputs = model(features.float())

        # Compute loss
        loss = criterion(outputs, labels)

        # Check for nan loss
        if torch.isnan(loss):
            print("Encountered nan loss during training")
            continue  # Skip this batch or handle NaNs as needed

        clip_optimizer.zero_grad()
        pcbm_optimizer.zero_grad()
        loss.backward()
        clip_optimizer.step()
        pcbm_optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# Function to evaluate the model
def evaluate(model, clip_model, test_loader, criterion, preprocess, device):
    model.eval()
    clip_model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = preprocess(inputs).to(device)
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass through CLIP model
            # features = clip_model(inputs)
            features = clip_model.module.encode_image(inputs)

            # Forward pass through PCBM
            outputs = model(features.float())

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Store predictions and labels
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    return avg_loss, accuracy

# Main function
def main(args):
    # Define the save directory
    save_dir = "/data/gpfs/projects/punim2103/joint_training"

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    #_, preprocess = get_model(args, backbone_name=args.backbone_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load('RN50', device)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model = torch.nn.DataParallel(clip_model).to(args.device)
    preprocess = transforms.ToTensor()

    # Load dataset
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)

    # Load concept bank
    concept_bank = pickle.load(open(args.concept_bank, 'rb'))
    concept_bank = ConceptBank(concept_bank, args.device)

    # Initialize models
    #clip_model, preprocess = get_model(args, backbone_name=args.backbone_name)
    clip_model = clip_model.to(args.device)

    pcbm = PosthocLinearCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=len(classes))
    pcbm = pcbm.to(args.device)

    # Define optimizer and loss function
    # Define optimizers for CLIP and PCBM
    clip_optimizer = torch.optim.Adam(clip_model.parameters(), lr=args.lr)
    pcbm_optimizer = torch.optim.SGD(pcbm.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    resolution = 224

    preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    # Training and evaluation loop
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(pcbm, clip_model, train_loader, criterion, clip_optimizer, pcbm_optimizer, preprocess, args.device)
        val_loss, val_accuracy = evaluate(pcbm, clip_model, test_loader, criterion, preprocess, args.device)

        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'clip_model_state_dict': clip_model.state_dict(),
        'pcbm_model_state_dict': pcbm.state_dict(),
    }, final_model_path)

    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    args = config()
    main(args)