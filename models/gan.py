# segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)
import cv2
import os
import numpy as np

# Function to overlay two images
def overlay_images(segmented_img, human_body_img):
   
    segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)

    segmented_img = cv2.resize(segmented_img, (human_body_img.shape[1], human_body_img.shape[0]))
    _, mask = cv2.threshold(segmented_img, 1, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(mask)
    segmented_img = cv2.bitwise_and(segmented_img, segmented_img, mask=mask)
    human_body_img = cv2.bitwise_and(human_body_img, human_body_img, mask=inverted_mask)
    combined_img = cv2.add(segmented_img, human_body_img)

    return combined_img



segmented_folder = ''


human_body_folder = ''


output_folder = ''


for segmented_image_name in os.listdir(segmented_folder):

    segmented_image_path = os.path.join(segmented_folder, segmented_image_name)
    segmented_image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)

    for human_body_image_name in os.listdir(human_body_folder):
      
        human_body_image_path = os.path.join(human_body_folder, human_body_image_name)
        human_body_image = cv2.imread(human_body_image_path)

      
        if (segmented_image).shape[:2] != (human_body_image).shape[:2] or (segmented_image).shape[2] != 3:
            continue  

  
        overlaid_image = overlay_images(segmented_image, human_body_image)

      
        output_image_name = f"{segmented_image_name.split('.')[0]}_{human_body_image_name}"
        output_image_path = os.path.join(output_folder, output_image_name)
        cv2.imwrite(output_image_path, overlaid_image)

        print(f"Saved overlaid image: {output_image_path}")
