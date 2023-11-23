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
from torch.cuda.amp import GradScaler, autocast

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
    parser.add_argument("--clip-learning-rate", default=1e-7, type=float)
    parser.add_argument("--pcbm-learning-rate", default=1e-7, type=float)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--warmup", type=int, default=1000, help="number of steps to warmup for")
    parser.add_argument('--last_num_ft', type=int, default=-1, help="number of layers to refine for clip")
    return parser.parse_args()

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model, classifier):
        super().__init__()
        self.clip_model = clip_model
        self.linear = classifier # 10 classes for CIFAR-10

    def forward(self, images):
        with torch.no_grad():  # Freeze CLIP's weights
            image_features = self.clip_model.encode_image(images)
        return self.linear(image_features)


# Function to evaluate the model
def evaluate(probe_model, test_loader, criterion, preprocess, device):
    probe_model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = preprocess(inputs).to(device)
            inputs, labels = inputs.to(device), labels.to(device)
    
            # Forward pass 
            outputs = probe_model(inputs)

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
    save_dir = "/data/gpfs/projects/punim2103/joint_training/finetuning_clip"
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Load CLIP model
    clip_model, preprocess = clip.load('RN50', device, jit=False) #Must set jit=False for training
    clip_model = clip_model.to(args.device)
    classifier = torch.load('/data/gpfs/projects/punim2103/new_attempt_4_classifier_model_full.pth', map_location=device)
    probe_model = CLIPLinearProbe(clip_model, classifier).to(device)
    convert_models_to_fp32(probe_model) 
    
    # CIFAR-10 class names to generate CLIP text
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Load Datasets
    train_dataset = datasets.CIFAR10(root="/data/gpfs/projects/punim2103/data", train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_dataset = datasets.CIFAR10(root="/data/gpfs/projects/punim2103/data", train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)


    # Define preprocessing suitable for CLIP 
    resolution = 224

    preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])


    # Define optimizers and criterion
    if args.last_num_ft == -1:
        clip_optimizer = torch.optim.SGD([{'params': probe_model.clip_model.parameters()},  # Parameters of the visual part
                                          {'params': probe_model.linear.parameters()}],  # Parameters of the linear layer
                                    lr=args.clip_learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        clip_optimizer = torch.optim.SGD([{'params': list(probe_model.clip_model.parameters())[-args.last_num_ft:]},  # Parameters of the visual part
                                          {'params': probe_model.linear.parameters()}],  # Parameters of the linear layer
                                    lr=args.clip_learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    

    criterion = torch.nn.CrossEntropyLoss().to(device)

    scaler = GradScaler()

    
    # Define step scheduler 
    total_steps = len(train_loader) * args.num_epochs
    clip_scheduler = cosine_lr(clip_optimizer, args.clip_learning_rate, args.warmup, total_steps)
    

    # Evaluate before training
    val_loss, val_accuracy = evaluate(probe_model, test_loader, criterion, preprocess, args.device)
    print(f"Evaluation before training: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


    #  Switch to train mode
    probe_model.train()
    
    # Training
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        batch = 0


        # Step 4: Training loop for CLIP model
        for inputs, labels in train_loader:
            num_batches_per_epoch = len(train_loader)
            step = num_batches_per_epoch * epoch + batch
            clip_scheduler(step)

            
            clip_optimizer.zero_grad()

            inputs = preprocess(inputs).to(device)
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            with autocast():           
                outputs = probe_model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(clip_optimizer)
            scaler.update()

            batch += 1

        print("Evaluating...")
        val_loss, val_accuracy = evaluate(probe_model, test_loader, criterion, preprocess, args.device)

        # Print epoch results
        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the final model 
    final_model_path = os.path.join(save_dir, f'final_model_finetuned_{args.last_num_ft}_cliplayers_{args.clip_learning_rate}_clip_lr_{args.pcbm_learning_rate}_pcbm_lr.pth')
    torch.save({
        'clip_model_state_dict': probe_model.clip_model.state_dict(),
        'linear_model_state_dict': probe_model.linear.state_dict(),
    }, final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    args = config()
    main(args)