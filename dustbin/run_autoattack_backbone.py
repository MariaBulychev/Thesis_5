#import sys
#sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
#sys.path.append('/data/gpfs/projects/punim2103/auto_attack')  # Adding the auto_attack directory
#import sys
#sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm') 

#from autoattack.autoattack import AutoAttack  # This should work without problems now
from autoattack import AutoAttack

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import clip

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



backbone, preprocess = clip.load('RN50', device=args.device, download_root=args.out_dir)
backbone = backbone.eval()

class ClipImageEncoder(nn.Module):
    def __init__(self, clip_model, num_classes=10):
        super(ClipImageEncoder, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, x):
        embeddings = self.clip_model.encode_image(x)
        return self.classifier(embeddings)


clip_image_encoder = ClipImageEncoder(backbone).to(args.device)  # Ensure model is on the correct device
clip_image_encoder.eval()  # Ensure model is in evaluation mode

## Load the data
train_loader, test_loader, idx_to_class, classes = get_dataset(args, preprocess)

## Evaluate
adversary = AutoAttack(clip_image_encoder, norm='Linf', eps=0.3, version='standard')


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