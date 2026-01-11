import os

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from rembg import remove

from Feature_Extraction.ROI_Extraction import ROI_Extraction
from Sub_Functions.Read_data import Preprocessing, Feature_Extraction

class_names1 = {0: "Healthy Papaya", 1: "Anthracnose", 2: "Phytophthora Blight", 3: "Mechanical Damage",
                4: "Chocolate Spot", 5: "Stick Disease", 6: "Physiological Spot", 7: "Black Spot", 8: "Scar"}

import cv2
import numpy as np


def add_fitting_text(img, text, max_font_scale=1.3, color=(255, 255, 255), thickness=1,
                     bg_color=(0, 0, 0), padding=10, bg_opacity=0.7):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_height, img_width = img.shape[:2]

    # Target text area (90% width, 20% height, centered)
    target_width = img_width * 0.85
    target_height = img_height * 0.18

    best_scale = 0.3
    best_text_width, best_text_height = 0, 0

    for scale in np.linspace(0.3, max_font_scale, 50):
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        total_height = text_height + baseline + padding * 2

        if text_width <= target_width and total_height <= target_height:
            best_scale = scale
            best_text_width, best_text_height = text_width, text_height + baseline

    # Final text size calculation
    (text_width, text_height), baseline = cv2.getTextSize(text, font, best_scale, thickness)
    text_total_height = text_height + baseline

    # Background rectangle position (centered)
    bg_x1 = (img_width - text_width) // 2 - padding
    bg_y1 = (img_height - text_total_height) // 2 - padding // 2
    bg_x2 = bg_x1 + text_width + padding * 2
    bg_y2 = bg_y1 + text_total_height + padding

    # Ensure rectangle stays within image bounds
    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(img_width, bg_x2)
    bg_y2 = min(img_height, bg_y2)

    # Create semi-transparent black background
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)

    # Blend background with opacity
    alpha = bg_opacity
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Add white text on black background
    text_x = (img_width - text_width) // 2
    text_y = bg_y1 + padding + text_height

    cv2.putText(img, text, (text_x, text_y), font, best_scale, color, thickness)

    return img

def NEW_ROI(img):
    removed_bg = remove(img)
    removed_bg = cv2.cvtColor(removed_bg, cv2.COLOR_BGRA2RGB)

    gray = cv2.cvtColor(removed_bg, cv2.COLOR_RGB2GRAY)

    # 2️⃣ Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3️⃣ Binary threshold (black background → 0)
    _, binary = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

    # 4️⃣ Morphological close (fill small holes)
    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5️⃣ Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img   # fallback

    # 6️⃣ Largest contour = fruit
    largest = max(contours, key=cv2.contourArea)

    # 7️⃣ Create mask from contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    # 8️⃣ Apply mask to ORIGINAL RGB image
    fruit_rgb = cv2.bitwise_and(img, img, mask=mask)

    # 9️⃣ Tight crop using bounding box
    x, y, w, h = cv2.boundingRect(largest)
    fruit_rgb = fruit_rgb[y:y + h, x:x + w]

    return fruit_rgb


def Image_Results(folder_path, DB):
    global current_class
    images = os.listdir(folder_path)
    # i = 0
    for index, img in enumerate(images):
        splitted = img.split("-")
        if len(splitted) > 1 and img.endswith(".jpg"):
            img1 = os.path.join(folder_path,img)

            image = cv2.imread(img1)
            if splitted[1][0] == '0':
                current_class = "Healthy Papaya"
            if splitted[1][0] == '1':
                current_class = "Anthracnose"

            if splitted[1][0] == '2':
                current_class = "Phytophthora Blight"
            if splitted[1][0] == '3':
                current_class = "Mechanical Damage"
            if splitted[1][0] == '4':
                current_class = "Chocolate Spot"
            if splitted[1][0] == "5":
                current_class = "Stick Disease"

            if splitted[1][0] == "6":
                current_class = "Physiological Spot"
            if splitted[1][0] == "7":
                current_class = "Black Spot"
            if splitted[1][0] == "8":
                current_class = "Scar"

            # image = cv2.

            # images = os.listdir(complete_folder_path)
            # random1 = np.random.randint(0, 100, size=(2,))
            # random_images = [images[i] for i in random1]
            # for index, image in enumerate(random_images):
            #     image_path = os.path.join(complete_folder_path, image)

            # image = cv2.imread(image_path)
            image = cv2.resize(image, (250, 250))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image33 = image
            roi_out1 = NEW_ROI(image)

            gaussian, median = Preprocessing(roi_out1)

            pil_image = Image.open(img1).convert("RGB")

            pil_image = np.array(pil_image)

            roi_out2 = NEW_ROI(pil_image)

            roi_out2 = np.array(roi_out2)

            gaussian_pil, median_pil = Preprocessing(roi_out2)

            features_out = Feature_Extraction(gaussian, gaussian_pil)

            deep_map_based_feature = features_out[:, :, 0]

            Hybrid_depth_based_textural_pattern = features_out[:, :, 1]

            Modified_pixel_intensity_based_structural_pattern = features_out[:, :, 2]

            statistical_glcm_features = features_out[:, :, 3:]

            # current_class = class_names1[0]

            os.makedirs(f"Image_Results/{DB}/{current_class}/sample{index + 1}/", exist_ok=True)

            os.makedirs(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/", exist_ok=True)

            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/deep_map_based_feature.jpg",
                       deep_map_based_feature, )
            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/Hybrid_depth_based_textural_pattern.jpg",
                       Hybrid_depth_based_textural_pattern, cmap="grey")

            plt.imsave(
                f"Image_Results/{DB}/{current_class}/sample{index + 1}/Modified_pixel_intensity_based_structural_pattern.jpg",
                Modified_pixel_intensity_based_structural_pattern, cmap="grey")

            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/Energy.jpg",
                       statistical_glcm_features[:, :, 0], cmap="grey")
            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/dissimilarity.jpg",
                       statistical_glcm_features[:, :, 1], cmap="grey")
            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/contrast.jpg",
                       statistical_glcm_features[:, :, 2], cmap="grey")
            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/homogenity.jpg",
                       statistical_glcm_features[:, :, 3], cmap="grey")
            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/entropy.jpg",
                       statistical_glcm_features[:, :, 4], cmap="grey")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/Original.jpg", image33)
            plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/InceptionV3.jpg",
                       statistical_glcm_features[:, :, 5])
            median = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
            gaussian = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)

            cv2.imwrite(f"Image_Results/{DB}/{current_class}/sample{index + 1}/median_filter.jpg", median)
            cv2.imwrite(f"Image_Results/{DB}/{current_class}/sample{index + 1}/gaussian_filter.jpg", gaussian)

            output_image = add_fitting_text(image33, text=f"{current_class}")
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            roi_out1 = cv2.cvtColor(roi_out1, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"Image_Results/{DB}/{current_class}/sample{index + 1}/Output.jpg", output_image)
            cv2.imwrite(f"Image_Results/{DB}/{current_class}/sample{index + 1}/ROI_Extraction.jpg", roi_out1)
        # i = i + 1


Image_Results("Dataset/Dataset1/Train-20251230T164241Z-3-001/Train", "DB1")
