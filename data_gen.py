# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
#
# a = np.random.rand(100, 28, 28, 9)
# b = np.random.randint(0, 6, size=(100,))
#
# os.makedirs("data_loader/DB2/", exist_ok=True)
# np.save("data_loader/DB2/Features.npy", a)
# np.save("data_loader/DB2/Labels.npy", b)
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Create bounds directory
os.makedirs("Bounds/Dataset1", exist_ok=True)

img_bgr = cv2.imread(
    "Dataset/Dataset2/Papaya Diseases Dataset/Papaya Diseases Dataset/Papaya Disease/Papaya Disease/Anthracanose Diease/6.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

for i in range(50):  # Reduced range for speed (0-50 covers good range)
    for j in range(50):
        # Your bounds pattern
        lower_bound = np.array([10 + i, 10 + j, 10 + j], dtype=np.uint8)
        upper_bound = np.array([10 + j, 200 + i, 200 + i], dtype=np.uint8)

        # Create threshold mask using inRange (BGR space)
        mask = cv2.inRange(img_bgr, lower_bound, upper_bound)

        # Stats
        white_pixels = np.sum(mask == 255)

        # Plot
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title(f'Threshold Mask\nBounds: L[{lower_bound}] U[{upper_bound}]\nWhite pixels: {white_pixels}')

        # Save with your exact naming
        plt.savefig(f"Bounds/Dataset1/sample{i}{j}.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close instead of show() for speed

        if (i + 1) % 10 == 0 and j == 0:
            print(f"Processed i={i}, saved {i * 50 + j + 1} images...")

print("All threshold mask variations saved in Bounds/Dataset1/")
print("Filenames: sample{i}{j}.png with bounds in title!")

# -- here i need to find the best bounds by varying the bounds save the each images inside the Bounds/ apply fo rthe same image
# extract mask for same image and apply the different colors +5,+5 with each R,G,B and save the imaeg with the bound title , so that
# i can see the best bound , give me the code for it
