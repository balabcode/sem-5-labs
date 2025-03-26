# Write a program to obtain gradient of an image.

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)

gx_kernel = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])

gy_kernel = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]])

gx = cv2.filter2D(src=image, ddepth=-1, kernel=gx_kernel)
gy = cv2.filter2D(src=image, ddepth=-1, kernel=gy_kernel)

g_mag = np.sqrt(gx**2 + gy**2).astype(np.uint8)
g_dir = np.arctan2(gy, gx).astype(np.uint8)

images = (image, gx, gy, g_mag, g_dir)

cv2.imshow('images', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()