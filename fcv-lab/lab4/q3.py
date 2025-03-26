# Write a program to segment an image based on colour.

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png")

lower_bound = np.array([100, 0, 0])
upper_bound = np.array([160, 250, 250])

h,w,_ = image.shape

mask = np.zeros((h,w), dtype=np.uint8)

for y in range(h):
    for x in range(w):
        pixel = image[y,x]
        
        if (lower_bound <= pixel).all() and (pixel <= upper_bound).all():
            mask[y,x] = 255

segmented_img = cv2.bitwise_and(image, image, mask=mask)
images = (image, segmented_img)

cv2.imshow('all', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()