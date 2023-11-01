import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def eval_simple(model_path, broden_projections_path, labels_path, num_images=10):
    # Load CIFAR-10 test images
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Load the model
    model = torch.jit.load(model_path)
    model.eval()

    # Load broden projections and labels
    broden_projections = np.load(broden_projections_path)
    labels = np.load(labels_path)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(range(num_images)):  # Only take the first `num_images` images
            image, label = dataset[idx]
            image = image.unsqueeze(0)

            # Convert broden projection to tensor
            broden_proj = torch.tensor(broden_projections[idx]).unsqueeze(0).float()

            output = model(image, broden_proj)
            all_preds.append(output[0].detach().cpu().numpy())
            all_labels.append(label)

    all_preds = np.concatenate(all_preds, axis=0)

    # Compute metrics if labels are binary
    if max(all_labels) == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        print("AUC:", auc)
    else:
        accuracy = np.mean(np.argmax(all_preds, axis=1) == np.array(all_labels))
        print("Accuracy:", accuracy)
    
    return all_preds

data_root = "/data/gpfs/projects/punim2103/h-train_results/"
model_path = data_root + "RN50.pt"
broden_projections_path = data_root + "test-proj_cifar10__clip:RN50__broden_clip:RN50_0.npy"
labels_path = data_root + "test-lbls_cifar10__clip:RN50__broden_clip:RN50_0_lbls.npy"

outputs = eval_simple(model_path, broden_projections_path, labels_path, num_images=10)
print("Model Outputs for first 10 images:", outputs)
