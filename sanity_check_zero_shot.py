import sys
import clip
import torch
import torch.nn as nn
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoattack import AutoAttack
from torchvision.transforms import ToPILImage

class ModelWrapper(nn.Module):
    def __init__(self, clip_model, resolution):
        super(ModelWrapper, self).__init__()
        self.clip_model = clip_model
        self.preprocess = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        
        # CIFAR-10 class descriptions for zero-shot evaluation
        self.class_descriptions = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in self.class_descriptions]).to(device)

    def forward(self, images):
        images = self.preprocess(images)
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(self.text_inputs)
        
        # Pick the top class for each image
        similarity = (image_features @ text_features.T)
        #values, indices = similarity.max(dim=1)
        
        return similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
batch_size = 128

print("load model")
resolution = 224
wrapped_model = ModelWrapper(model, resolution).to(device)
wrapped_model.eval()

print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(1000))
test_loader = DataLoader(test_subset, batch_size=batch_size)

epsilon = float(sys.argv[1]) / 255.

adversary = AutoAttack(wrapped_model, norm='Linf', eps=epsilon, version='standard', device=device)

csv_path = f'/data/gpfs/projects/punim2103/csv_linf/original_model_sanity_check_zero_shot_results_eps_{epsilon}.csv'
batch = 0
results = []

with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Epsilon", "Initial Accuracy", "Robust Accuracy", "Max Perturbation"])  # Header

    for images, labels in test_loader:
        print("start attack for epsilon:", epsilon)
        batch += 1
        print("batch "+str(batch))
        
        images, labels = images.to(device), labels.to(device)

        outputs = wrapped_model(images)
        _, predicted = outputs.max(dim=1)

        initial_acc = (predicted == labels).sum().item() / images.shape[0]
        print(f'Initial Accuracy for Batch {batch}: {100 * initial_acc:.2f}%')

        x_adv, robust_accuracy, res = adversary.run_standard_evaluation(images, labels, bs=batch_size)

        #torch.save(x_adv, f'/data/gpfs/projects/punim2103/autoattack_results/l_inf/original_model_sanity_check_2/eps_{epsilon}_batch_{batch}_adv.pt')

        results.append([100 * initial_acc, 100* robust_accuracy, res.item()])
        csv_writer.writerow([epsilon, 100 * initial_acc, 100 * robust_accuracy, res.item()])

        print("done")
        if batch * batch_size >= 1000:
            break

    # Calculate and write the mean values at the end of the csv
    mean_values = [epsilon] + [sum(col)/len(col) for col in zip(*results)]
    csv_writer.writerow([])
    csv_writer.writerow(["Mean for Epsilon", "Mean Initial Accuracy", "Mean Robust Accuracy", "Mean Max Perturbation"])
    csv_writer.writerow(mean_values)
