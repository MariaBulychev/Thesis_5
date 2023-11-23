import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, get_model
from training_tools import load_or_compute_projections
import clip
from torchvision import datasets, transforms
from sklearn.linear_model import SGDClassifier
import torch.nn as nn

# Define the configuration for the model
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--num-epochs", default=30, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    return parser.parse_args()

class ModelWrapper(nn.Module):
    def __init__(self, pcbm, clip_model, resolution):
        super(ModelWrapper, self).__init__()
        self.pcbm = pcbm
        self.clip_model = clip_model
        
        # Define the preprocessing pipeline within the ModelWrapper
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        features = self.clip_model.encode_image(images)
        out = self.pcbm(features.float().to(args.device), return_dist = False)
        return out


# Function to evaluate the model
def evaluate(pcbm, clip_model, test_loader, criterion, preprocess, device):
    pcbm.eval()
    clip_model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            #inputs = preprocess(inputs).to(device)
            inputs, labels = inputs.to(device), labels.to(device)
    
            # Forward pass through PCBM
            features = clip_model.encode_image(inputs)
            features = features.float().to(device)
            #print("Features dtype:", features.float().to(device).dtype)
            #print("Features dtype in evaluate:", features.dtype)  # Debugging print

            # Check the data type of pcbm parameters
            #for name, param in pcbm.named_parameters():
                #print(f"{name} dtype: {param.dtype}")  # Debugging print

            outputs = pcbm(features)

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

    # Load CLIP model
    clip_model, preprocess = clip.load('RN50', device)
    clip_model = clip_model.to(args.device)
    clip_model.visual.train()
    #for param in clip_model.parameters():
        #param.requires_grad = False
    
    
    # Load dataset
    #preprocess = transforms.ToTensor()
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)

    # Load concept bank
    concept_bank = pickle.load(open(args.concept_bank, 'rb'))
    concept_bank = ConceptBank(concept_bank, args.device)

    # Load PCBM    
    pcbm = torch.load('/data/gpfs/projects/punim2103/train_results/pcbm_cifar10__clip:RN50__broden_clip:RN50_0__lam:0.0002__alpha:0.99__seed:42.ckpt', map_location=device)

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_accuracy = evaluate(pcbm, clip_model, test_loader, criterion, preprocess, args.device)

    # Print epoch results
    print(f"Evaluation before training: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    #optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(pcbm.trainable_params(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}/{args.num_epochs}')

        

        # Step 4: Training loop for CLIP model
        for inputs, labels in train_loader:
            # inputs = preprocess(inputs).to(device)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass through PCBM
            features = clip_model.encode_image(inputs)
            features = features.float().to(device)
            out = pcbm(features)

            # Backpropagation
            
            loss = criterion(out, labels)
            loss.backward()        
            optimizer.step()

        print("Evaluating...")
        val_loss, val_accuracy = evaluate(pcbm, clip_model, test_loader, criterion, preprocess, args.device)

        # Print epoch results
        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

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