import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append('/data/gpfs/projects/punim2103')
from post_hoc_cbm.data.data_zoo import get_dataset
from post_hoc_cbm.models import get_model, PosthocLinearCBM, PosthocHybridCBM


# Define a mock-args class to replicate the argparse object
class MockArgs:
    def __init__(self):
        self.dataset = 'cifar10'
        self.out_dir = './data'  # Modify to where you want the dataset to be downloaded
        self.backbone_name = "clip:RN50"
        self.batch_size = 64
        self.num_workers = 4  # Number of CPU processes for data loading
        self.distributed = False
        self.device = "cuda"

args = MockArgs()
backbone, preprocess = get_model(args, backbone_name=args.backbone_name)

# Load the trained model
# model_path = '/data/gpfs/projects/punim2103/h-train_results/RN50.pt'
# model = torch.load(model_path)

#checkpoint_path = "/data/gpfs/projects/punim2103/train_results/pcbm-hybrid_cifar10__clip:RN50__broden_clip:RN50_0__lam:0.0002__alpha:0.99__seed:42.ckpt"
#checkpoint = torch.load(checkpoint_path)
#model = PosthocHybridCBM.load_state_dict(checkpoint['state_dict'])

model_path = "/data/gpfs/projects/punim2103/train_results/pcbm-hybrid_cifar10__clip:RN50__broden_clip:RN50_0__lam:0.0002__alpha:0.99__seed:42.ckpt"
model = torch.load(model_path)

# Load the data
test_embeddings = np.load('/data/gpfs/projects/punim2103/h-train_results/test-embs_cifar10__clip:RN50__broden_clip:RN50_0.npy')
test_projections = np.load('/data/gpfs/projects/punim2103/h-train_results/test-proj_cifar10__clip:RN50__broden_clip:RN50_0.npy')
test_labels = np.load('/data/gpfs/projects/punim2103/h-train_results/test-lbls_cifar10__clip:RN50__broden_clip:RN50_0_lbls.npy')

# Convert data to tensors
test_embeddings = torch.tensor(test_embeddings)
test_projections = torch.tensor(test_projections)
test_labels = torch.tensor(test_labels)

print("emb: "+str(test_embeddings.shape))
#print("emb 0: "+str(test_embeddings[0]))
print("proj: "+str(test_projections.shape))
#print("proj 0: "+str(test_projections[0]))


train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)

print(" bis hier gehts") 
tqdm_loader = tqdm(test_loader)

# Evaluate
model.eval()  # Set the model to evaluation mode

test_loader = DataLoader(TensorDataset(torch.tensor(test_embeddings).float(), torch.tensor(test_labels).long()), batch_size=args.batch_size, shuffle=False)

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
        #test_embeddings = test_embeddings.long().to(args.device)
        outputs = model(batch_X)#, test_embeddings)
        print(outputs[0])
        print(outputs.shape)
        #outputs = model(test_embeddings)

        print("Print Concepts")
        # Prints the Top-5 Concept Weigths for each class.
        print(model.bottleneck.analyze_classifier(k=5))

    # Now you can calculate any metric you want using `outputs` and `test_labels`








