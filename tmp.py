from PIL import Image
import matplotlib.pyplot as plt

# Replace 'your_image.png' with the actual file name of the image you want to open
image_path = '/home/mbulychev/pcbm/data/broden_concepts/air_conditioner/positives/ADE_train_00001091.png'
save_path = '/data/gpfs/projects/punim2103/tmp/2008_006403.png'

# Open the image
image = Image.open(image_path)

image.save(save_path)
