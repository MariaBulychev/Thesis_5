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
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=int, default=1)
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

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, criterion, X, target, alpha, attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    upper_limit, lower_limit = 1, 0
    delta = torch.zeros_like(X).cuda()

    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)

    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon

    else:
        raise ValueError
    
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    for _ in range(attack_iters):
        # output = model(normalize(X ))

        output = model(X+delta)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]

        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)

        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)

        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


# Combined model for generating adversarial examples 
class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model, pcbm):
        super().__init__()
        self.clip_model = clip_model
        self.pcbm = pcbm # 10 classes for CIFAR-10

        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        images = self.preprocess(images)
        image_features = self.clip_model.encode_image(images)
        return self.pcbm(image_features.float())


# Function to evaluate the model
def evaluate(clip_model, pcbm, test_loader, criterion, preprocess, device):
    clip_model.eval()
    pcbm.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = preprocess(inputs).to(device)
            inputs, labels = inputs.to(device), labels.to(device)
    
            # Forward pass through PCBM
            features = clip_model.encode_image(inputs)
            outputs = pcbm(features.float().to(args.device), return_dist = False)

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
    save_dir = f"/data/gpfs/projects/punim2103/joint_training/finetuning_both_{args.num_epochs}_epochs_different_lr"
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Load CLIP model
    clip_model, preprocess = clip.load('RN50', device, jit=False) #Must set jit=False for training
    clip_model = clip_model.to(args.device)
    convert_models_to_fp32(clip_model) 
    


    # Load PCBM
    pcbm = torch.load('/data/gpfs/projects/punim2103/train_results/pcbm_cifar10__clip:RN50__broden_clip:RN50_0__lam:0.0002__alpha:0.99__seed:42.ckpt', map_location=device)
    convert_models_to_fp32(pcbm) 
    
    for param in pcbm.trainable_params():
        param.requires_grad = True


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
        clip_optimizer = torch.optim.SGD(clip_model.visual.parameters(),
                                    lr=args.clip_learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        clip_optimizer = torch.optim.SGD(list(clip_model.visual.parameters())[-args.last_num_ft:],
                                    lr=args.clip_learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
    pcbm_optimizer = torch.optim.SGD(pcbm.trainable_params(),
                                    lr=args.pcbm_learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)    

    criterion = torch.nn.CrossEntropyLoss().to(device)

    scaler = GradScaler()

    
    # Define step scheduler 
    total_steps = len(train_loader) * args.num_epochs
    clip_scheduler = cosine_lr(clip_optimizer, args.clip_learning_rate, args.warmup, total_steps)
    pcbm_scheduler = cosine_lr(pcbm_optimizer, args.pcbm_learning_rate, args.warmup, total_steps)
    

    # Evaluate before training
    val_loss, val_accuracy = evaluate(clip_model, pcbm, test_loader, criterion, preprocess, args.device)
    print(f"Evaluation before training: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


    #  Switch to train mode
    clip_model.visual.train()
    pcbm.train()
    
    # Training
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        batch = 0
        attack_norm = 'l_inf'
        

        # Step 4: Training loop for CLIP model
        for inputs, labels in train_loader:
            num_batches_per_epoch = len(train_loader)
            step = num_batches_per_epoch * epoch + batch
            clip_scheduler(step)
            pcbm_scheduler(step)

            
            clip_optimizer.zero_grad()
            pcbm_optimizer.zero_grad()

            
            inputs, labels = inputs.to(device), labels.to(device)

            
            with autocast():   
                # Calculate original outputs
                orig_images = preprocess(inputs).to(device)
                orig_features = clip_model.encode_image(orig_images)
                orig_features = orig_features.to(device) #features.float().to(device)
                outputs_orig = pcbm(orig_features)

                #### Incorporate adversarial examples ####
                # Generate adversarial examples
                probe_model = CLIPLinearProbe(clip_model, pcbm).to(device)
                convert_models_to_fp32(probe_model) 
                # inputs go in unpreprocessed because probe_model does preprocessing 
                delta = attack_pgd(probe_model, criterion, inputs, labels, alpha=args.train_stepsize, attack_iters=args.train_numsteps, norm=attack_norm, restarts=1, early_stop=True, epsilon=0.001)       
                
                # preprocess after the attack (because now we dont evaluate with the probe_model)               
                adv_images = inputs + delta
                adv_images = preprocess(adv_images).to(device)
                
                # Calculate adversarial outputs
                adv_features = clip_model.encode_image(adv_images)
                adv_features = adv_features.to(device) #features.float().to(device)
                outputs_adv = pcbm(adv_features)

                total_outputs = outputs_orig + outputs_adv
                loss = criterion(total_outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(clip_optimizer)
            scaler.step(pcbm_optimizer)
            scaler.update()

            batch += 1

        print("Evaluating...")
        val_loss, val_accuracy = evaluate(clip_model, pcbm, test_loader, criterion, preprocess, args.device)

        # Print epoch results
        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the final model
    final_model_path = os.path.join(save_dir, f'final_model_finetuned_{args.last_num_ft}_cliplayers_{args.clip_learning_rate}_clip_lr_{args.pcbm_learning_rate}_pcbm_lr.pth')
    torch.save({
        #'wrapped_model_state_dict': wrapped_model.state_dict(),
        'clip_model_state_dict': clip_model.state_dict(),
        'pcbm_model_state_dict': pcbm.state_dict(),
    }, final_model_path)

    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    args = config()
    main(args)