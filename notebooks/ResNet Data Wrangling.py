# Databricks notebook source
import os
from pathlib import Path
from collections import defaultdict
from shutil import copy
from pprint import pprint

import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2, InterpolationMode
print(torch.__version__)

# COMMAND ----------

# dbutils.fs.ls("/Volumes/catalog1/schema1/data1")
data_path = Path("/Volumes/catalog1/schema1/data1/food-101")

image_path = data_path / "images"
meta_path = data_path / "meta"

train_path = data_path / "train"
test_path = data_path / "test"

model_path = data_path / 'model_data'

# COMMAND ----------

# def walk_through_dir(dir_path):
#     for dirpath, dirnames, filenames in os.walk(dir_path):
#         print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# walk_through_dir(image_path)

# COMMAND ----------

# # Helper method to split dataset into train and test folders
# def prepare_data(filepath, src, dest):
#     classes_images = defaultdict(list)
#     with open(filepath, 'r') as txt:
#         paths = [read.strip() for read in txt.readlines()]
#         for p in paths:
#             food = p.split('/')
#             classes_images[food[0]].append(food[1] + '.jpg')

#     for food in classes_images.keys():
#         print("\nCopying images into ",food)
#         if not os.path.exists(os.path.join(dest,food)):
#             os.makedirs(os.path.join(dest,food))
#         for i in classes_images[food]:
#             copy(os.path.join(src,food,i), os.path.join(dest,food,i))
#     print("Copying Done!")

# COMMAND ----------

# prepare_data(meta_path / 'train.txt', image_path, train_path)
# prepare_data(meta_path / 'test.txt', image_path, test_path)

# COMMAND ----------

# DBTITLE 1,Visualize random image
image_path_list = list(image_path.glob("*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem
# img = Image.open(random_image_path).convert(mode='RGB', palette=Image.Palette.ADAPTIVE).quantize(colors=16)
img = Image.open(random_image_path).convert(mode='RGB')

print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
img

# COMMAND ----------

# import numpy as np

# img_as_array = np.asarray(img)

# plt.figure(figsize=(8, 4))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} \n Image shape: {img_as_array.shape} --> [height, width, color_channels]")
# plt.axis(False);

# COMMAND ----------

# DBTITLE 1,Construct an Torchvision Transform
width = 224
height = 224

data_transform = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(size=(232, 232), interpolation=InterpolationMode.BILINEAR),
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# COMMAND ----------

def plot_transformed_images(image_paths: Path, transform: v2.Compose, n: int = 3, seed: int = 83) -> None:
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k = n)

    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            # Column 1
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nsize: {f.size}")
            ax[0].axis("off")

            # Column 2
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nsize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

# COMMAND ----------

plot_transformed_images(image_path_list, transform=data_transform, n=3)

# COMMAND ----------


