#import sys
#sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
#sys.path.append('/data/gpfs/projects/punim2103/auto_attack')  # Adding the auto_attack directory


#from autoattack.autoattack import AutoAttack  # This should work without problems now
from autoattack import AutoAttack

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

#import sys
#sys.path.append('../')

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

#print(sys.path)

model_path = "/data/gpfs/projects/punim2103/train_results/pcbm-hybrid_cifar10__clip:RN50__broden_clip:RN50_0__lam:0.0002__alpha:0.99__seed:42.ckpt"
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode



## Load the data
backbone, preprocess = get_model(args, backbone_name=args.backbone_name)
backbone.eval()
train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)

# (maybe not needed) Load embeddings and projections 
test_embeddings = np.load('/data/gpfs/projects/punim2103/h-train_results/test-embs_cifar10__clip:RN50__broden_clip:RN50_0.npy')
test_projections = np.load('/data/gpfs/projects/punim2103/h-train_results/test-proj_cifar10__clip:RN50__broden_clip:RN50_0.npy')
test_labels = np.load('/data/gpfs/projects/punim2103/h-train_results/test-lbls_cifar10__clip:RN50__broden_clip:RN50_0_lbls.npy')

# (maybe not needed) Convert data to tensors
test_embeddings = torch.tensor(test_embeddings)
test_projections = torch.tensor(test_projections)
test_labels = torch.tensor(test_labels)


## Define wrapper function as the model takes embeddings as input, while autoattack need the original image 
#def wrapper_model_fn(batch_X):  #backbone, model

    #embeddings = backbone.encode_image(batch_X).detach().float() # works only for CLIP models
    #logits = model(embeddings)

    #return logits


    

class AdversarialModel(nn.Module):
    def __init__(self, backbone, model):
        super(AdversarialModel, self).__init__()
        self.backbone = backbone
        self.model = model

    def get_embeddings(self, x):
        return self.backbone.encode_image(x).float()  #.detach().float()  # encoding using the backbone

    def forward(self, x):
        if x.dim() == 4:  # Assuming x is an image tensor with shape [batch, channels, height, width]
            x = self.get_embeddings(x)
        logits = self.model(x)  # passing the output of the backbone or embeddings through the model
        return logits

#adversarial_model = AdversarialModel(backbone, model)


adversarial_model = AdversarialModel(backbone, model).to(args.device)
adversarial_model.eval()  # set to evaluation mode


## Evaluate
adversary = AutoAttack(adversarial_model, norm='Linf', eps=0.3, version='standard')

# test_loader = DataLoader(TensorDataset(torch.tensor(test_embeddings).float(), torch.tensor(test_labels).long()), batch_size=args.batch_size, shuffle=False)

with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
        #test_embeddings = test_embeddings.long().to(args.device)
        #outputs = model(batch_X)#, test_embeddings)
        adv_images = adversary.run_standard_evaluation(batch_X, batch_Y, bs=batch_X.size(0))
        #torch.save(adv_images, 'adv_images.pt')

        #print(adv_images)
        #print(adv_images.shape)
        #print(outputs[0])
        #print(outputs.shape)

        #print("Print Concepts")
        # Prints the Top-5 Concept Weigths for each class.
        #print(model.bottleneck.analyze_classifier(k=5))

    # Now you can calculate any metric you want using `outputs` and `test_labels`