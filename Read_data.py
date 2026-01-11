import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from Feature_Extraction.Deep_map_based_feature import Deep_map_based_features
from Feature_Extraction.Hybrid_Depth_based_textural_pattern import Hybrid_depth_based_textural_pattern
from Feature_Extraction.Modified_pixel_intensity_based_structural_pattern import \
    Modified_pixel_intensity_based_structural_pattern
from Feature_Extraction.Statistical_GLCM_Features import statistical_glcm_features
from Feature_Extraction.Inception_v3 import Inception_model
from Feature_Extraction.ROI_Extraction import ROI_Extraction


def ROI_E(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([25, 40, 40])
    upper_white = np.array([90, 255, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    fruit_mask = cv2.bitwise_not(white_mask)

    kernel = np.ones((5, 5), np.uint8)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    padding = 0

    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    cropped_img = img[y:y + h, x:x + w]
    cropped_img = cv2.resize(cropped_img, (250, 250))
    return cropped_img


def Preprocessing(img):
    median = cv2.medianBlur(img, 5)
    Gaussian = cv2.GaussianBlur(median, (5, 5), 0)

    return Gaussian ,median


def Feature_Extraction(img, pil_image):
    DMBF = Deep_map_based_features(img)
    HDBTP = Hybrid_depth_based_textural_pattern(pil_image)

    MPIBSP = Modified_pixel_intensity_based_structural_pattern(img)
    SGF = statistical_glcm_features(img)
    Inceptionv3 = Inception_model(img)

    DMBF = np.expand_dims(DMBF, axis=2)
    HDBTP = np.expand_dims(HDBTP, axis=2)
    MPIBSP = np.expand_dims(MPIBSP, axis=2)
    # SGF = np.expand_dims(SGF, axis=2)
    inception = np.expand_dims(Inceptionv3, axis=2)

    features = np.concatenate([DMBF, HDBTP, MPIBSP, SGF, inception], axis=2)

    return features


def Read_data(DB):
    import os
    from PIL import Image
    import cv2
    import numpy as np

    if DB == "DB2":

        folder_path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease"

        class_names = os.listdir(folder_path)
        features_final = []
        labels_final = []
        i = 0
        for class_name in class_names:

            complete_folder_path = os.path.join(folder_path, class_name)

            images = os.listdir(complete_folder_path)[:1]
            for image in images:
                image_path = os.path.join(complete_folder_path, image)
                image = cv2.imread(image_path)

                roi_out1 = ROI_Extraction(image)

                preprocessed_out = Preprocessing(roi_out1)

                pil_image = Image.open(image_path).convert("RGB")

                pil_image = np.array(pil_image)

                roi_out2 = ROI_Extraction(pil_image)

                roi_out2 = np.array(roi_out2)

                preprocessed_out_pil = Preprocessing(roi_out2)

                features_out = Feature_Extraction(preprocessed_out, preprocessed_out_pil)

                features_final.append(features_out)

                labels_final.append(i)

            i = i + 1

        os.makedirs("data_loader/DB2/", exist_ok=True)
        np.save("data_loader/DB2/Features.npy", np.array(features_final))
        np.save("data_loader/DB2/Labels.npy", np.array(labels_final))

    import os
    import cv2
    import numpy as np
    from PIL import Image
    from collections import Counter

    # Class mapping from Sisfrutos-Papaya (adjust if you have exact list from data/papaya-data.yaml)
    CLASS_CODES = {
        0: 'Papaya', 1: 'Anthracnose', 2: 'Black Spot', 3: 'Ring Spot',
        4: 'Chocolate Spot', 5: 'Phytophthora', 6: 'Powdery Mildew',
        7: 'Scar', 8: 'Healthy'
    }

    def parse_filename_labels(filename):
        """Extract class codes from filename like 'TT000582-8-7-7.jpg' -> [8,7,7]"""
        if '-' not in filename or filename.endswith('.jpg'):
            base = filename.replace('.jpg', '')
        else:
            base = filename
        parts = base.split('-')
        codes = []
        for part in parts[2:]:  # Skip take C-C...
            if part.isdigit():
                codes.append(int(part))
        return list(set(codes))  # Unique classes per image

    def parse_txt_labels(txt_path):
        """Parse .txt annotations for class IDs: class x y w h per line"""
        classes = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    classes.append(class_id)
        return list(set(classes))  # Unique classes

    if DB == "DB1":
        folder_path = "Dataset/Dataset1/Train-20251230T065023Z-3-001/Train"
        features_final = []
        labels_final = []  # Now populated with class lists or primary class

        content = os.listdir(folder_path)
        images = [i for i in content if i.endswith('.jpg')]

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            txt_path = os.path.join(folder_path, img_name.replace('.jpg', '.txt'))

            # Extract labels from filename AND txt (prioritize txt for accuracy)
            filename_labels = parse_filename_labels(img_name)
            txt_labels = parse_txt_labels(txt_path)
            all_labels = list(set(filename_labels + txt_labels))  # Combined unique classes

            # For classification: use primary/dominant class (most common) or multi-label
            primary_label = Counter(all_labels).most_common(1)[0][0] if all_labels else 8  # Default healthy

            # Process image (your existing pipeline)
            img1 = cv2.imread(img_path)
            roi_out1 = ROI_E(img1)
            preprocessed_out = Preprocessing(roi_out1)

            pil_image = Image.open(img_path).convert("RGB")
            pil_image = np.array(pil_image)
            roi_out2 = ROI_Extraction(pil_image)
            roi_out2 = np.array(roi_out2)
            preprocessed_out_pil = Preprocessing(roi_out2)

            features_out = Feature_Extraction(preprocessed_out, preprocessed_out_pil)
            features_final.append(features_out)
            labels_final.append(primary_label)  # Or all_labels for multi-label

        os.makedirs("data_loader/DB2/", exist_ok=True)
        np.save("data_loader/DB2/Features.npy", np.array(features_final))
        np.save("data_loader/DB2/Labels.npy", np.array(labels_final))
        print(f"Processed {len(images)} images. Label distribution: {Counter(labels_final)}")
