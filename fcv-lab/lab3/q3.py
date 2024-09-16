# Write a program to compare box filter and gaussian filter image outputs.

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png")

gaussian_size = 5
gaussian_kernel = np.zeros((gaussian_size, gaussian_size), np.float32)
center = gaussian_size // 2
sum_val = 0

for x in range(gaussian_size):
    for y in range(gaussian_size):
        delta_x = x - center
        delta_y = y - center
        value = (1 / (2 * np.pi)) * np.exp(-((delta_x**2 + delta_y**2) / (2)))
        gaussian_kernel[x, y] = value
        sum_val += value

gaussian_kernel /= sum_val


box_kernel = np.ones((3, 3), np.float32) / 9

box_filtered = cv2.filter2D(image, -1, box_kernel)
gaussian_filtered = cv2.filter2D(image, -1, gaussian_kernel)

images = (image, box_filtered, gaussian_filtered)

cv2.imshow('all', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()