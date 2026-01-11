# # import cv2
# # import numpy as np
# # import pandas as pd
# # import os
# # from PIL import Image
# # from Feature_Extraction.Deep_map_based_feature import Deep_map_based_features
# # from Feature_Extraction.Hybrid_Depth_based_textural_pattern import Hybrid_depth_based_textural_pattern
# # from Feature_Extraction.Modified_pixel_intensity_based_structural_pattern import \
# #     Modified_pixel_intensity_based_structural_pattern
# # from Feature_Extraction.Statistical_GLCM_Features import statistical_glcm_features
# # from Feature_Extraction.Inception_v3 import Inception_model
# # from Feature_Extraction.ROI_Extraction import ROI_Extraction
# # from Sub_Functions.Read_data import Feature_Extraction, Preprocessing
# # from keras.models import load_model, Model
# # import matplotlib.pyplot as plt
# #
# # img_path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/Anthracanose Diease/1.tif"
# # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
# # if img is None:
# #     raise ValueError(f"Failed to load {img_path}")
# #
# # print(f"Original image shape: {img.shape}")
# #
# # pr = Preprocessing(img)
# # roi_out1 = ROI_Extraction(pr)
# # preprocessed_out = Preprocessing(roi_out1)
# #
# # features_out = Feature_Extraction(preprocessed_out, preprocessed_out)
# #
# # sample_shape = features_out.reshape(features_out.shape[0] * features_out.shape[1], features_out.shape[2])
# # plt.imsave("Feature_verify/Multiplied_height_width.jpg", sample_shape)
# # print(f"Raw features_out shape: {features_out.shape}")
# #
# # features_out = features_out.reshape(features_out.shape[0],
# #                                     features_out.shape[1] * features_out.shape[2])
# #
# # plt.imsave("Feature_con.jpg", features_out)
# # print(f"Reshaped features_out: {features_out.shape}")
# #
# # model = load_model("Saved_models/DB2.h5")
# # model.summary()
# # new_model1 = Model(inputs=model.input, outputs=model.get_layer("bidirectional").output)
# #
# # new_model2 = Model(inputs=model.input, outputs=model.get_layer("bidirectional_6").output)
# #
# # inp = np.expand_dims(features_out, axis=0)
# #
# # print(f"Model input shape: {inp.shape}")
# #
# # out = new_model1.predict(inp, verbose=0)
# # out2 = np.squeeze(out)
# # mm = new_model2.predict(inp)
# # mm2 = np.squeeze(out)
# # plt.imsave("Feature_verify/bidirectional_5.jpg", mm2)
# # plt.imshow(out2)
# # plt.imsave("Feature_verify/Bidirectionalstm_1.jpg", out2)
# # plt.show()
# #
# # print(f"Model output shape: {out.shape}")
# # print(f"Prediction: {out}")
# #
# import os
#
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
    count1 = np.random.randint(0, 100, size=(40,))
    images = [images[count] for count in count1]
    for index, image in enumerate(images):
        image_path = os.path.join(complete_folder_path, image)
        img1 = cv2.imread(image_path)

        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([95, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # upper_gray = np.array([40, 50, 22])
        img1_no_green = img1.copy()
        img1_no_green[green_mask > 0] = (0, 0, 0)

        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (250, 250))

        th_value = 60
        # if class_name == "Good Papaya":
        #     _, mask_global = cv2.threshold(gray, 190, 235, cv2.THRESH_BINARY)
        # else:
        _, mask_global = cv2.threshold(gray, 190, 235, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

        lower_fruit = np.array([15, 40, 40], dtype=np.uint8)
        upper_fruit = np.array([95, 255, 255], dtype=np.uint8)

        fruit_mask = cv2.inRange(hsv, lower_fruit, upper_fruit)
        fruit_mask = cv2.resize(fruit_mask, (250, 250), interpolation=cv2.INTER_NEAREST)

        background_mask = cv2.bitwise_not(fruit_mask)

        mask_global[background_mask > 0] = 0

        white_pixels = np.sum(mask_global == 255)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img1 = cv2.resize(img1, (250, 250))
        # plt.imshow(img1, cmap="gray")
        # plt.axis("off")

        plt.subplot(1, 2, 1)
        img_show = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img_show = cv2.resize(img_show, (250, 250))
        plt.imshow(img_show)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"mask(white pixel count= {white_pixels}")
        plt.imshow(mask_global, cmap="gray")
        plt.axis("off")
        os.makedirs(f"Gray_sample/{class_name}", exist_ok=True)
        plt.savefig(f"Gray_sample/{class_name}/sample{index}.jpg")
        # plt.pause(1)
        plt.close()

# ok FCM NOT Works properly lets go with thresholding , listen i need the cdein the same old flow for saving, 2 subplots
# in a single image and here i need the image like extract the disease region ,create a mask and extract the disease region as white and
# others black , the disease region is in different colors , i upload sample 4 images in that image in the diseases region ther
# is some colors note that bounds from 4 images and use it in my code and give me the code crrectly to extarct all diseased region
# and save the mask image in the same format how i have saved now with 2 subplots with white pixel counts in the top

# here in the all the images the third one is the good papaya , give me the color bounds for all diseases correctly to extract the diseases region exactlyin the mask like now in the code . with accuracte color bounding
# # mask_adapt = cv2.adaptiveThreshold(
# #     gray,
# #     255,
# #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #     cv2.THRESH_BINARY_INV,
# #     35,
# #     5
# # )
# #
# # plt.figure(figsize=(8, 4))
# # plt.subplot(1, 2, 1)
# # plt.title("Gray")
# # plt.imshow(gray, cmap="gray")
# # plt.axis("off")
# # plt.subplot(1, 2, 2)
# # plt.title(f"Adaptive mask")
# # plt.imshow(mask_adapt, cmap="gray")
# # plt.axis("off")
# # plt.show()


# import os
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
# folder_path = "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease"
#
# class_names = os.listdir(folder_path)
# features_final = []
# labels_final = []
#
# for class_name in class_names:
#     i = 0
#     complete_folder_path = os.path.join(folder_path, class_name)
#
#     images = os.listdir(complete_folder_path)
#     count1 = np.random.randint(0, 100, size=(10,))
#     images = [images[count] for count in count1]
#
#     for index, image in enumerate(images):
#         image_path = os.path.join(complete_folder_path, image)
#         img1 = cv2.imread(image_path)
#
#         hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
#         hsv = cv2.resize(hsv, (250, 250))
#
#         lower_bound = np.array([20, 20, 30,])
#         upper_bound = np.array([180, 245, 255])
#
#         mask = cv2.inRange(hsv, lower_bound, upper_bound)
#
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#         mask = cv2.GaussianBlur(mask, (5, 5), 0)
#
#         white_pixels = np.sum(mask == 255)
#         total_pixels = mask.shape[0] * mask.shape[1]
#         white_percentage = (white_pixels / total_pixels) * 100
#
#         plt.figure(figsize=(8, 4))
#
#         plt.subplot(1, 2, 1)
#         img1 = cv2.resize(img1, (250, 250))
#         img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#         img_rgb = cv2.resize(img_rgb, (250, 250))
#         plt.imshow(img_rgb)
#         plt.title(f"Original")
#         plt.axis("off")
#
#         plt.subplot(1, 2, 2)
#         plt.title(f"Mask (White pixels: {white_pixels})")
#         plt.imshow(mask, cmap="gray")
#         plt.axis("off")
#
#         os.makedirs(f"Gray_sample/{class_name}", exist_ok=True)
#         plt.savefig(f"Gray_sample/{class_name}/sample{index}_mask.jpg")
#         plt.pause(100)
#         plt.close()
#
#         result = cv2.bitwise_and(img1, img1, mask=mask)
#         cv2.imwrite(f"Gray_sample/{class_name}/result_{index}.jpg", result)

