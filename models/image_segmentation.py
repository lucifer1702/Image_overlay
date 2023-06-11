# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

# !pip install datasets

from datasets import load_dataset

# Load the Fashion ControlNet dataset
dataset = load_dataset('abrumu/fashion_controlnet_dataset')

print(dataset.keys())

train_data=dataset['train']
type(train_data)

import os 
import shutil

# !mkdir dataset

from PIL import Image

num_samples = len(train_data)
output_dir = "/content/dataset"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(num_samples):
    sample = train_data[i]
    # sample = train_data[i]
    image_array = sample['target']
    image_array = np.array(image_array)
    # print(img_array)
    image = Image.fromarray(image_array)
    image_filename = f"image_{i}.png"
    output_image_path = os.path.join(output_dir, image_filename)
    image.save(output_image_path)

# !mkdir output_final

import torch
import torchvision.transforms as transforms
from torchvision import models

# Load the pre-trained DeepLab v3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define the transformation to preprocess the image
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


input_folder = "/content/dataset"
output_folder = "/content/output_final"


for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert("RGB")

        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Run the image through the model
        with torch.no_grad():
            output = model(input_batch)["out"][0]
        output_predictions = output.argmax(0)

        # Convert the predicted labels to a binary mask
        mask = output_predictions.byte().cpu().numpy()

        # Apply the mask to the original image
        segmented_image = image.copy()
        segmented_image.putalpha(Image.fromarray(mask * 255))

        # Save the segmented image to the output folder
        output_path = os.path.join(output_folder, filename)
        segmented_image.save(output_path)

        print(f"Segmented image saved at: {output_path}")

# !zip -r output_final.zip /content/output_final

# !zip -r dataset.zip /content/dataset