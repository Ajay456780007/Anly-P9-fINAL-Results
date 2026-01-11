import numpy as np
from rembg import remove
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Dataset/Dataset1/Train-20251230T065023Z-3-001/Train/TR000000-7.jpg")

new_img = remove(img)

plt.subplot(1, 5, 1)
new_img_rgba = cv2.cvtColor(new_img, cv2.COLOR_BGRA2RGB)
# new_img = cv2.cvtColor(new_img, cv2.COLOR_RGBA2RGB)
hsv = cv2.cvtColor(new_img_rgba, cv2.COLOR_BGR2HSV)
lower_color = np.array([50, 60, 30])
upper_color = np.array([131, 235, 230])

mask1 = cv2.inRange(hsv, (8,60,40), (30,255,220))
mask2 = cv2.inRange(hsv, (0,80,50), (10,255,230))
disease_mask = cv2.bitwise_or(mask1, mask2)


out = cv2.inRange(new_img_rgba, lower_color, upper_color)

threshold = 160
new_img_gray = cv2.cvtColor(new_img_rgba, cv2.COLOR_RGB2GRAY)
_, thres = cv2.threshold(new_img_gray, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)
plt.imshow(new_img_rgba)

plt.subplot(1, 5, 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.subplot(1, 5, 3)
plt.imshow(thres, cmap="gray")
plt.subplot(1, 5, 4)
plt.imshow(out, cmap="gray")
plt.subplot(1,5,5)
plt.imshow(disease_mask,cmap="gray")
plt.show()
