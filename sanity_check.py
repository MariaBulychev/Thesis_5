import sys
import clip
import torch
import torch.nn as nn
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from autoattack import AutoAttack
from torchvision.transforms import ToPILImage
from resnet import ResNet18



device = "cuda" if torch.cuda.is_available() else "cpu"


print("load model")

model = ResNet18()
ckpt = torch.load("/data/gpfs/projects/punim2103/pretrained88.pth", map_location=device)
model.load_state_dict(ckpt)
model.to(device)
#model.cuda()
model.eval()



batch_size = 128



print("load data")
test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_subset = Subset(test_dataset, range(10000))
test_loader = DataLoader(test_subset, batch_size=batch_size)

epsilon = float(sys.argv[1]) / 255.

adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', device=device)

csv_path = f'/data/gpfs/projects/punim2103/csv_sanity_check/resnet_results_eps_{epsilon}.csv'
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

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        initial_acc = (predicted == labels).sum().item() / images.shape[0]
        print(f'Initial Accuracy for Batch {batch}: {100 * initial_acc:.2f}%')

        x_adv, robust_accuracy, res = adversary.run_standard_evaluation(images, labels, bs=batch_size)

        #torch.save(x_adv, f'/data/gpfs/projects/punim2103/autoattack_results/sanity_check/eps_{epsilon}_batch_{batch}_adv.pt')

        results.append([100 * initial_acc, 100* robust_accuracy, res.item()])
        csv_writer.writerow([epsilon, 100 * initial_acc, 100 * robust_accuracy, res.item()])

        print("done")
        if batch * batch_size >= 10000:
            break

    # Calculate and write the mean values at the end of the csv
    mean_values = [epsilon] + [sum(col)/len(col) for col in zip(*results)]
    csv_writer.writerow([])
    csv_writer.writerow(["Mean for Epsilon", "Mean Initial Accuracy", "Mean Robust Accuracy", "Mean Max Perturbation"])
    csv_writer.writerow(mean_values)
