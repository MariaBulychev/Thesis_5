import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import clip
from torchvision import datasets, transforms

from data import get_dataset
from concepts import ConceptBank
from models import PosthocLinearCBM, PosthocHybridCBM, get_model
from training_tools import load_or_compute_projections_ft, AverageMeter, MetricComputer




def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=str, help="Output folder")
    parser.add_argument("--checkpoint-path", required=True, type=str, help="Trained PCBM module.")
    #parser.add_argument("--pcbm-path", required=True, type=str, help="Trained PCBM module.")
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank.")
    parser.add_argument("--backbone-name", required=True, type=str, help="To save embs.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--dataset", default="cub", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num-epochs", default=20, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--l2-penalty", default=0.001, type=float)
    return parser.parse_args()


@torch.no_grad()
def eval_model(args, posthoc_layer, loader, num_classes):
    epoch_summary = {"Accuracy": AverageMeter()}
    tqdm_loader = tqdm(loader)
    computer = MetricComputer(n_classes=num_classes)
    all_preds = []
    all_labels = []
    
    for batch_X, batch_Y in tqdm(loader):
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device) 
        out = posthoc_layer(batch_X)            
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())
        metrics = computer(out, batch_Y) 
        epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0]) 
        summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
        summary_text = "Eval - " + " ".join(summary_text)
        tqdm_loader.set_description(summary_text)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        return auc
    return epoch_summary["Accuracy"]


def train_hybrid(args, train_loader, val_loader, posthoc_layer, optimizer, num_classes):
    cls_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.num_epochs+1):
        print(f"Epoch: {epoch}")
        epoch_summary = {"CELoss": AverageMeter(),
                         "Accuracy": AverageMeter()}
        tqdm_loader = tqdm(train_loader)
        computer = MetricComputer(n_classes=num_classes)
        for batch_X, batch_Y in tqdm(train_loader):
            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            optimizer.zero_grad()
            out, projections = posthoc_layer(batch_X, return_dist=True)
            cls_loss = cls_criterion(out, batch_Y)
            loss = cls_loss + args.l2_penalty*(posthoc_layer.residual_classifier.weight**2).mean()
            loss.backward()
            optimizer.step()
            
            epoch_summary["CELoss"].update(cls_loss.detach().item(), batch_X.shape[0])
            metrics = computer(out, batch_Y) 
            epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0])

            summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
            summary_text = " ".join(summary_text)
            tqdm_loader.set_description(summary_text)
        
        latest_info = dict()
        latest_info["epoch"] = epoch
        latest_info["args"] = args
        latest_info["train_acc"] = epoch_summary["Accuracy"]
        latest_info["test_acc"] = eval_model(args, posthoc_layer, val_loader, num_classes)
        print("Final test acc: ", latest_info["test_acc"])
    return latest_info



def main(args, pcbm, clip_model, preprocess):
    
    train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess=preprocess) # own preprocessing defined in main
    num_classes = len(classes)
    
    hybrid_model_path = args.checkpoint_path.replace("final_model_", "pcbm-hybrid_")
    run_info_file = hybrid_model_path.replace("pcbm", "run_info-pcbm")
    run_info_file = run_info_file.replace(".ckpt", ".pkl")
    
    run_info_file = os.path.join(args.out_dir, run_info_file)
    
    # We use the precomputed embeddings and projections.
    train_embs, _, train_lbls, test_embs, _, test_lbls = load_or_compute_projections_ft(args, clip_model, pcbm, train_loader, test_loader) # own function to ensure that projections are computed again

    
    train_loader = DataLoader(TensorDataset(torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()), batch_size=args.batch_size, shuffle=False)

    # Initialize PCBM-h
    hybrid_model = PosthocHybridCBM(pcbm) # CLIP ist hier nicht drin weil wir das neue clip schon benutzt haben um die neuen projections zu berechnen
    hybrid_model = hybrid_model.to(args.device)
    
    # Initialize the optimizer
    hybrid_optimizer = torch.optim.Adam(hybrid_model.residual_classifier.parameters(), lr=args.lr)
    hybrid_model.residual_classifier = hybrid_model.residual_classifier.float()
    hybrid_model.bottleneck = hybrid_model.bottleneck.float() # bottleneck = pcbm
    
    # Train PCBM-h
    run_info = train_hybrid(args, train_loader, test_loader, hybrid_model, hybrid_optimizer, num_classes)

    torch.save(hybrid_model, hybrid_model_path)
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)
    
    print(f"Saved to {hybrid_model_path}, {run_info_file}")

if __name__ == "__main__":    
    args = config()    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the PCBM
    # pcbm path is now the path to the checkpoint of the combined model 
    #checkpoint_path = torch.load(args.checkpoint_path,)

    #checkpoint_path = '/data/gpfs/projects/punim2103/joint_training/finetuning_only_clip_adv_20_epochs/final_model_finetuned_-1_cliplayers_1e-07_clip_lr_1e-07_pcbm_lr.pth'
    checkpoint = torch.load(args.checkpoint_path, map_location = device)
    # Load the CLIP model state_dict
    clip_model_state_dict = checkpoint['clip_model_state_dict']
    # Load the PCBM model state_dict
    linear_model_state_dict = checkpoint['pcbm_model_state_dict']

    clip_model, preprocess = clip.load('RN50', device)
    clip_model.load_state_dict(clip_model_state_dict)
    #batch_size = 128

    pcbm = torch.load('/data/gpfs/projects/punim2103/train_results/pcbm_cifar10__clip:RN50__broden_clip:RN50_0__lam:0.0002__alpha:0.99__seed:42.ckpt', map_location=device)
    pcbm.load_state_dict(linear_model_state_dict)
    resolution = 224
    
    clip_model = clip_model.to(args.device)
    pcbm = pcbm.to(args.device)

    clip_model.eval()
    pcbm.eval()

    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    main(args, pcbm, clip_model, preprocess)
