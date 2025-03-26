# Harris Corner Detection

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)

BLOCK_SIZE = 2

def find_gradient(image):
    x_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    y_kernel = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    gx = cv2.filter2D(image, -1, x_kernel)
    gy = cv2.filter2D(image, -1, y_kernel)

    g_mag = np.sqrt(gx**2 + gy**2)
    g_dir = np.arctan2(gy, gx) * 180.0 / np.pi

    return g_mag, g_dir, gx, gy

g_mag, g_dir, gx, gy = find_gradient(image)

det = ((gx**2) * (gy**2)) - ((gx*gy)**2)
trace = ((gx**2) + (gy**2))

l = 0.04
R = np.abs(det - l*(trace ** 2))

max_val = R.max()
corners = np.zeros_like(image)
corners[R > 0.01 * max_val] = 255


images = (image, corners)
cv2.imshow('images', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()