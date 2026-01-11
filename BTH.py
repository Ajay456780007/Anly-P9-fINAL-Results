import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

folder_path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease"

class_names = os.listdir(folder_path)
features_final = []
labels_final = []

for class_name in class_names:
    i = 0
    complete_folder_path = os.path.join(folder_path, class_name)

    images = os.listdir(complete_folder_path)
    count1 = np.random.randint(0, min(100, len(images)), size=(40,))
    images = [images[count] for count in count1 if count < len(images)]

    for index, image in enumerate(images):
        image_path = os.path.join(complete_folder_path, image)
        img1 = cv2.imread(image_path)
        if img1 is None:
            continue

        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

        # Remove background green
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        img1_no_green = img1.copy()
        img1_no_green[green_mask > 0] = (0, 0, 0)

        # Resize for processing
        hsv_resized = cv2.resize(hsv, (250, 250))
        gray = cv2.cvtColor(cv2.resize(img1_no_green, (250, 250)), cv2.COLOR_BGR2GRAY)

        # Disease-specific color masks (tuned for your 4 sample images)
        # Brown/dark spots [0-30H, 50-255S, 0-100V]
        lower_disease1 = np.array([0, 50, 0], dtype="uint8")
        upper_disease1 = np.array([30, 255, 100], dtype="uint8")

        # Orange/yellow diseased spots [15-35H, 100-255S, 80-200V]
        lower_disease2 = np.array([15, 100, 80], dtype="uint8")
        upper_disease2 = np.array([35, 255, 200], dtype="uint8")

        # Black/very dark spots [any H, low S/V]
        lower_disease3 = np.array([0, 0, 0], dtype="uint8")
        upper_disease3 = np.array([180, 100, 60], dtype="uint8")

        mask_disease1 = cv2.inRange(hsv_resized, lower_disease1, upper_disease1)
        mask_disease2 = cv2.inRange(hsv_resized, lower_disease2, upper_disease2)
        mask_disease3 = cv2.inRange(hsv_resized, lower_disease3, upper_disease3)

        # Combine all disease regions
        mask_global = cv2.bitwise_or(mask_disease1, mask_disease2)
        mask_global = cv2.bitwise_or(mask_global, mask_disease3)

        # Restrict to fruit region only
        lower_fruit = np.array([15, 40, 40], dtype=np.uint8)
        upper_fruit = np.array([95, 255, 255], dtype=np.uint8)
        fruit_mask = cv2.inRange(hsv_resized, lower_fruit, upper_fruit)
        fruit_mask = cv2.resize(fruit_mask, (250, 250), interpolation=cv2.INTER_NEAREST)
        background_mask = cv2.bitwise_not(fruit_mask)
        mask_global[background_mask > 0] = 0

        # Clean up mask with morphology
        kernel = np.ones((3, 3), np.uint8)
        mask_global = cv2.morphologyEx(mask_global, cv2.MORPH_CLOSE, kernel)
        mask_global = cv2.morphologyEx(mask_global, cv2.MORPH_OPEN, kernel)

        white_pixels = np.sum(mask_global == 255)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        img_show = cv2.cvtColor(cv2.resize(img1, (250, 250)), cv2.COLOR_BGR2RGB)
        plt.imshow(img_show)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"mask (white pixel count= {white_pixels})")
        plt.imshow(mask_global, cmap="gray")
        plt.axis("off")

        os.makedirs(f"Gray_sample/{class_name}", exist_ok=True)
        plt.savefig(f"Gray_sample/{class_name}/sample{index}.jpg")
        plt.close()

        # Optional: collect features
        # area_ratio = white_pixels / (250*250)
        # features_final.append([area_ratio])
        # labels_final.append(class_names.index(class_name))
