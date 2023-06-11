import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
segmented_folder = "/content/output_final"
normal_folder = "/content/input"
output_folder = "/content/match"
similarity_threshold = 0.1

for normal_filename in os.listdir(normal_folder):
    if normal_filename.endswith(('.jpg', '.jpeg', '.png')):
        
        normal_image_path = os.path.join(normal_folder, normal_filename)
        normal_image = cv2.imread(normal_image_path, cv2.IMREAD_GRAYSCALE)

        if normal_image is not None:
            
            fitting_segmented_images = []

            
            for segmented_filename in os.listdir(segmented_folder):
                if segmented_filename.endswith(('.jpg', '.jpeg', '.png')):
                   
                        segmented_image_path = os.path.join(segmented_folder, segmented_filename)
                        if segmented_image.shape[0] <= normal_image.shape[0] and segmented_image.shape[1] <= normal_image.shape[1]:
                            # Compute the structural similarity index
                            similarity_index = ssim(normal_image, segmented_image)

                            
                            if similarity_index >= similarity_threshold:
                                fitting_segmented_images.append(segmented_image)

            
            if fitting_segmented_images:
                output_subfolder = os.path.join(output_folder, normal_filename)
                os.makedirs(output_subfolder, exist_ok=True)

                for idx, segmented_image in enumerate(fitting_segmented_images):
                    output_path = os.path.join(output_subfolder, f"{idx}.png")
                    cv2.imwrite(output_path, segmented_image)

                print(f"Fitting segmented images saved for {normal_filename}")
            else:
                print(f"No fitting segmented images found for {normal_filename}")
        else:
            print(f"Error loading normal image: {normal_filename}")

