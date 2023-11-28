import sys
import clip
import torch
import torch.nn as nn
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('/data/gpfs/projects/punim2103')          # Adding the main project directory
sys.path.append('/data/gpfs/projects/punim2103/post_hoc_cbm')
from autoattack import AutoAttack
from torchvision.transforms import ToPILImage

class ModelWrapper(nn.Module):
    def __init__(self, classifier, clip_model, resolution):
        super(ModelWrapper, self).__init__()
        self.classifier = classifier
        self.clip_model = clip_model
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, images):
        images = self.preprocess(images)
        features = self.clip_model.encode_image(images)
        outputs = self.classifier(features.float().to(device))
        return outputs

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = '/data/gpfs/projects/punim2103/joint_training/finetuning_only_clip_adv_and_orig_20_epochs/final_model_finetuned_-1_cliplayers_1e-07_clip_lr_1e-07_pcbm_lr.pth'
checkpoint = torch.load(checkpoint_path, map_location = device)
# Load the CLIP model state_dict
clip_model_state_dict = checkpoint['clip_model_state_dict']
# Load the PCBM model state_dict
linear_model_state_dict = checkpoint['linear_model_state_dict']

model, preprocess = clip.load('RN50', device)
model.load_state_dict(clip_model_state_dict)
batch_size = 128

classifier = torch.load('/data/gpfs/projects/punim2103/new_attempt_4_classifier_model_full.pth', map_location=device)
classifier.load_state_dict(linear_model_state_dict)
resolution = 224
wrapped_model = ModelWrapper(classifier, model, resolution).to(device)
wrapped_model.eval()

print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(1000))
test_loader = DataLoader(test_subset, batch_size=batch_size)

epsilon = float(sys.argv[1])

adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard', device=device)

csv_path = f'/data/gpfs/projects/punim2103/results_clean/finetuned_models/only_clip_20_epochs_adv_and_orig/csv/eps_{epsilon}.csv'
batch = 0
results = []

with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Epsilon", "Initial Accuracy", "Robust Accuracy", "Max Perturbation"])  # Header

    for images, labels in test_loader:
        batch += 1
        print("batch "+str(batch))
        #if batch <= 42:  # Skip the first 58 batches
            #continue
        
        images, labels = images.to(device), labels.to(device)

        outputs = wrapped_model(images)
        _, predicted = torch.max(outputs, 1)
        initial_acc = (predicted == labels).sum().item() / images.shape[0]
        print(f'Initial Accuracy for Batch {batch}: {100 * initial_acc:.2f}%')

        x_adv, robust_accuracy, res = adversary.run_standard_evaluation(images, labels, bs=images.shape[0])

        torch.save(x_adv, f'/data/gpfs/projects/punim2103/results_clean/finetuned_models/only_clip_20_epochs_adv_and_orig/images/eps_{epsilon}_batch_{batch}_adv.pt')

        results.append([100 * initial_acc, 100* robust_accuracy, res.item()])
        csv_writer.writerow([epsilon, 100 * initial_acc, 100 * robust_accuracy, res.item()])

        print("done")
        

    # Calculate and write the mean values at the end of the csv
    mean_values = [epsilon] + [sum(col)/len(col) for col in zip(*results)]
    csv_writer.writerow([])
    csv_writer.writerow(["Mean for Epsilon", "Mean Initial Accuracy", "Mean Robust Accuracy", "Mean Max Perturbation"])
    csv_writer.writerow(mean_values)
