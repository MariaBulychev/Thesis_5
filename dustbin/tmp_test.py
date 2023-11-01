import torch
'''
# Define the paths to the saved tensors
tensor_path_original = "/data/gpfs/projects/punim2103/result_adv_images/original_tensors.pt"
tensor_path_adv = "/data/gpfs/projects/punim2103/result_adv_images/adversarial_tensors.pt"

# Load the tensors from the specified paths
original_images = torch.load(tensor_path_original)
adv_images = torch.load(tensor_path_adv)

# Extract the second image from the tensors
second_original_image = original_images[1]  # 0-indexed, so 1 is the second image
second_adv_image = adv_images[1]

# Compute the range of pixel values for the second original image
min_pixel_value_original = torch.min(second_original_image)
max_pixel_value_original = torch.max(second_original_image)

# Compute the range of pixel values for the second adversarial image
min_pixel_value_adv = torch.min(second_adv_image)
max_pixel_value_adv = torch.max(second_adv_image)

# Print the results
print("Range of pixel values for the second original image:", min_pixel_value_original.item(), "to", max_pixel_value_original.item())
print("Range of pixel values for the second adversarial image:", min_pixel_value_adv.item(), "to", max_pixel_value_adv.item())
'''

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

# Define the paths to the saved tensors
tensor_path_original = "/data/gpfs/projects/punim2103/result_adv_images/original_tensors.pt"
tensor_path_adv = "/data/gpfs/projects/punim2103/result_adv_images/adversarial_tensors.pt"

# Load tensors
orig = torch.load(tensor_path_original)
adv = torch.load(tensor_path_adv)

# Calculate the differences
differences = orig - adv

# Compute the L-infinity norm for each image
linf_norms = differences.view(16, -1).abs().max(dim=1).values

print(linf_norms)