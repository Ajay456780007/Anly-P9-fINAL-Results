# import os
#
# import cv2
# import numpy as np
# import pandas as pd
#
# import matplotlib.pyplot as plt
# from PIL import Image
#
# from Feature_Extraction.ROI_Extraction import ROI_Extraction
# from Sub_Functions.Read_data import Preprocessing, Feature_Extraction
#
# class_names1 = ["Anthracanose Diease", "Black Spot Diease", "Good Papaya", "phytophthora Disease",
#                 "Powdery Mildery Diease", "Ring spot Diease"]
#
# import cv2
# import numpy as np
#
#
# def add_fitting_text(img, text, max_font_scale=20.0, color=(255, 255, 255), thickness=2):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#
#     # Calculate text size at max scale
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, max_font_scale, thickness)
#
#     # Scale down if too wide/tall
#     img_height, img_width = img.shape[:2]
#     scale_w = (img_width * 0.9) / text_width
#     scale_h = (img_height * 0.15) / text_height
#     font_scale = min(max_font_scale, scale_w, scale_h, 0.7)  # min 0.01
#
#     # Recalculate final size
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#
#     # Center position
#     x = (img_width - text_width) // 2
#     y = (img_height + text_height) // 2  # Middle or adjust: 30 (top), img_height-30 (bottom)
#
#     cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
#     return img
#
#
# def Image_Results(folder_path, DB):
#     class_names = os.listdir(folder_path)
#     i = 0
#     for class_name in class_names:
#         complete_folder_path = os.path.join(folder_path, class_name)
#
#         images = os.listdir(complete_folder_path)
#         random1 = np.random.randint(0, 100, size=(2,))
#         random_images = [images[i] for i in random1]
#         for index, image in enumerate(random_images):
#             image_path = os.path.join(complete_folder_path, image)
#
#             image = cv2.imread(image_path)
#             image = cv2.resize(image, (250, 250))
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image33 = image
#             roi_out1 = ROI_Extraction(image)
#
#             gaussian, median = Preprocessing(roi_out1)
#
#             pil_image = Image.open(image_path).convert("RGB")
#
#             pil_image = np.array(pil_image)
#
#             roi_out2 = ROI_Extraction(pil_image)
#
#             roi_out2 = np.array(roi_out2)
#
#             gaussian_pil,median_pil = Preprocessing(roi_out2)
#
#             features_out = Feature_Extraction(gaussian, gaussian_pil)
#
#             deep_map_based_feature = features_out[:, :, :3]
#
#             Hybrid_depth_based_textural_pattern = features_out[:, :, 1]
#
#             Modified_pixel_intensity_based_structural_pattern = features_out[:, :, 2]
#
#             statistical_glcm_features = features_out[:, :, 3:]
#
#             current_class = class_names1[i]
#
#             os.makedirs(f"Image_Results/{DB}/{current_class}/sample{index + 1}/", exist_ok=True)
#
#             os.makedirs(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/", exist_ok=True)
#
#             plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/deep_map_based_feature.jpg",
#                        deep_map_based_feature, )
#         #     plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/Hybrid_depth_based_textural_pattern.jpg",
#         #                Hybrid_depth_based_textural_pattern,cmap="grey")
#         #
#         #     plt.imsave(
#         #         f"Image_Results/{DB}/{current_class}/sample{index + 1}/Modified_pixel_intensity_based_structural_pattern.jpg",
#         #         Modified_pixel_intensity_based_structural_pattern, cmap="grey")
#         #
#         #     plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/Energy.jpg",
#         #                statistical_glcm_features[:, :, 0], cmap="grey")
#         #     plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/dissimilarity.jpg",
#         #                statistical_glcm_features[:, :, 1], cmap="grey")
#         #     plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/contrast.jpg",
#         #                statistical_glcm_features[:, :, 2], cmap="grey")
#         #     plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/homogenity.jpg",
#         #                statistical_glcm_features[:, :, 3], cmap="grey")
#         #     plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/GLCM_Features/entropy.jpg",
#         #                statistical_glcm_features[:, :, 4], cmap="grey")
#         #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         #     plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/Original.jpg", image33)
#         #     plt.imsave(f"Image_Results/{DB}/{current_class}/sample{index + 1}/InceptionV3.jpg",
#         #                statistical_glcm_features[:, :, 5])
#         #     median = cv2.cvtColor(median,cv2.COLOR_BGR2RGB)
#         #     gaussian = cv2.cvtColor(gaussian,cv2.COLOR_BGR2RGB)
#         #
#         #     cv2.imwrite(f"Image_Results/{DB}/{current_class}/sample{index + 1}/median_filter.jpg",median)
#         #     cv2.imwrite(f"Image_Results/{DB}/{current_class}/sample{index + 1}/gaussian_filter.jpg",gaussian)
#         #
#         #     output_image = add_fitting_text(image33, text=f"{current_class}")
#         #     output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
#         #     roi_out1 = cv2.cvtColor(roi_out1, cv2.COLOR_BGR2RGB)
#         #     cv2.imwrite(f"Image_Results/{DB}/{current_class}/sample{index + 1}/Output.jpg", output_image)
#         #     cv2.imwrite(f"Image_Results/{DB}/{current_class}/sample{index + 1}/ROI_Extraction.jpg", roi_out1)
#         i = i + 1
#
#
# Image_Results("Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease", "DB2")

import cv2
import matplotlib.pyplot as plt

# img = cv2.imread("Dataset/Dataset1/Train-20251230T065023Z-3-001/Train/TR000002-8-8-8.jpg")
#
# # YOLO normalized values (center_x, center_y, width, height)
# x_norm, y_norm, w_norm, h_norm = 0.412906, 0.742927, 0.348375, 0.147601
#
# h, w, _ = img.shape
#
# # Convert YOLO → pixel corners
# x1 = int((x_norm - w_norm / 2) * w)
# y1 = int((y_norm - h_norm / 2) * h)
# x2 = int((x_norm + w_norm / 2) * w)
# y2 = int((y_norm + h_norm / 2) * h)
#
# # Draw rectangle
# cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# # Convert BGR → RGB for matplotlib
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# plt.imshow(img)
# plt.show()

import numpy as np
import pandas as pd
import os

# folder_path = "Dataset/Dataset1/Train-20251230T065023Z-3-001/Train/"
# count = 0
# single = 0
# content = os.listdir(folder_path)
# for i in content:
#     if i.endswith(".jpg"):
#         parts = i.split("-")
#         if len(parts) > 2:
#             count = count + 1
#         else:
#             single = single + 1
#
# print("Single class image:", single)
# print("Multiclass image:", count)

# import cv2
#
# path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/Anthracanose Diease"
# images = os.listdir(path)
# for i in range(len(images)):
#     image_path = os.path.join(path, images[i])
#     img = cv2.imread(image_path)
#     img = cv2.resize(img,(250,250))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     threshold = 90
#
#     _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
#     out1 = np.sum(out==255)
#     plt.subplot(1, 2, 1)
#     plt.imshow(out, cmap="gray")
#     plt.title(f"white pixels:{out1}")
#     plt.axis("off")
#     plt.subplot(1, 2, 2)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.axis("off")
#     os.makedirs("Bounds/Dataset1/Anthracanose Diease/",exist_ok=True)
#     plt.savefig(f"Bounds/Dataset1/Anthracanose Diease/sample{i}.png")
#     # plt.pause(1)
#     plt.close()


# import cv2
#
# path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/Black Spot Diease"
# images = os.listdir(path)
# for i in range(len(images)):
#     image_path = os.path.join(path, images[i])
#     img = cv2.imread(image_path)
#     img = cv2.resize(img,(250,250))
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     threshold = 80
#
#     _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
#     out1 = np.sum(out == 255)
#     plt.subplot(1, 2, 1)
#     plt.imshow(out, cmap="gray")
#     plt.title(f"White pixels:{out1}")
#     plt.axis("off")
#     plt.subplot(1, 2, 2)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.axis("off")
#     os.makedirs("Bounds/Dataset1/Black Spot Diease/", exist_ok=True)
#     plt.savefig(f"Bounds/Dataset1/Black Spot Diease/sample{i}.png")
#     plt.pause(1)
#     plt.close()

#
import cv2

#
path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/Good Papaya"
images = os.listdir(path)
for i in range(len(images)):
    image_path = os.path.join(path, images[i])uuu
    img = cv2.imread(image_path)
    img = cv2.resize(img, (250, 250))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # canny_edge = cv2.Canny(gray,threshold1=50,threshold2=150)
    # plt.imshow(canny_edge)
    # plt.show()
    lower_ywllow = np.array([10,90,150])
    upper_yellow = np.array([40,225,245])

    map = cv2.inRange(gray,lower_ywllow,upper_yellow)

    # _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    out1 = np.sum(map == 255)
    plt.subplot(1, 2, 1)
    plt.imshow(map, cmap="gray")
    plt.title(f"White pixels:{out1}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    os.makedirs("Bounds/Dataset1/Good Papaya1/", exist_ok=True)
    plt.savefig(f"Bounds/Dataset1/Good Papaya1/sample{i}.png")
    # plt.pause(1)
    plt.close()


# import cv2
#
# from rembg import remove
# path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/phytophthora Disease"
# images = os.listdir(path)
# for i in range(len(images)):
#     image_path = os.path.join(path, images[i])
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (250, 250))
#
#     bg_removed = remove(img)
#     gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGRA2GRAY)

# canny = cv2.Canny(gray, threshold1=20, threshold2=150)


# canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
# for i in range(canny.shape[0] - 2):
#     for j in range(canny.shape[1] - 2):
#         if (canny[i, j] == [255, 255, 255]).all():
#             canny[i][j - 1] = (255, 0, 0)
#             break
# for ii in reversed(range(canny.shape[0] - 2)):
#     for jj in reversed(range(canny.shape[1] - 2)):
#         if (canny[ii, jj] == [255,0,0]).all():
#             canny[ii][jj + 1] = ()
#             break

# for i in range(canny.shape[0] - 2):
#     for j in range(canny.shape[1] - 2):
#         if (canny[i, j] == [255, 255, 255]).all():
#             canny[i][j - 1] = (255, 0, 0)
#             break
#
# for ii in reversed(range(canny.shape[0] - 2)):
#     for jj in reversed(range(canny.shape[1] - 2)):
#         if (canny[ii, jj] == [255, 255, 255]).all():
#             canny[ii][jj + 1] = (255, 0, 0)
#             br
# threshold = 150
#
# _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
# out1 = np.sum(out == 255)
# plt.subplot(1, 2, 1)
# plt.imshow(out, cmap="gray")
# plt.title(f"White pixels:{out1}")
# plt.axis("off")
# plt.subplot(1, 2, 2)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.axis("off")
# os.makedirs("Bounds/Dataset1/phytophthora Disease/", exist_ok=True)
# plt.savefig(f"Bounds/Dataset1/phytophthora Disease/sample{i}.png")
# plt.pause(1)
# plt.close()

#
import cv2

from rembg import remove
path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/Ring spot Diease"
images = os.listdir(path)
for i in range(len(images)):
    image_path = os.path.join(path, images[i])
    img = cv2.imread(image_path)
    img = cv2.resize(img, (250, 250))
    bg_removed = remove(img)
    gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)

    threshold = 140

    _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    out1 = np.sum(out == 255)
    plt.subplot(1, 2, 1)
    plt.imshow(out, cmap="gray")
    plt.title(f"White pixels:{out1}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    os.makedirs("Bounds/Dataset1/Ring spot Diease/", exist_ok=True)
    plt.savefig(f"Bounds/Dataset1/Ring spot Diease/sample{i}.png")
    # plt.pause(1)
    plt.close()


# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/phytophthora Disease/"
# images = os.listdir(path)
# for i in range(len(images)):
#     image_path = os.path.join(path, images[i])
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (250, 250))
#
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     lower_white = np.array([0, 0, 160])  # ignore pure gray background by keeping V high
#     upper_white = np.array([180, 80, 255])
#
#     # 2) Pink / light‑red rot
#     lower_pink = np.array([160, 40, 80])
#     upper_pink = np.array([179, 255, 255])
#
#     # 3) Blue‑green / bluish rot spots
#     lower_bluegreen = np.array([80, 30, 40])
#     upper_bluegreen = np.array([120, 255, 220])
#
#     mask_white = cv2.inRange(hsv, lower_white, upper_white)
#     mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
#     mask_bluegr = cv2.inRange(hsv, lower_bluegreen, upper_bluegreen)
#
#     # final disease mask (union of all components)
#     mask = mask_white | mask_pink | mask_bluegr
#
#     # small noise removal
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     out1 = np.sum(mask == 255)
#     plt.subplot(1, 2, 1)
#     plt.imshow(mask, cmap="gray")
#     plt.title(f"White pixels:{out1}")
#     plt.axis("off")
#     plt.subplot(1, 2, 2)
#     img_with_black_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_with_black_bg)
#     plt.axis("off")
#
#     os.makedirs("Bounds/Dataset1/phytophthora Disease/", exist_ok=True)
#     plt.savefig(f"Bounds/Dataset1/phytophthora Disease/sample{i}.png")
#     plt.pause(1)
#     plt.close()

# here instaed of threshold i need to use the bounds upper and lower bounds i need to ignore the grey and white area
# and little yellow region in the image all others are good regions , give me the correct code for it
