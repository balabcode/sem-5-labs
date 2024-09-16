# Implement Canny edge detection algorithm.

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

g_mag = np.sqrt(gx**2 + gy**2)
g_dir = (np.arctan2(gy, gx) * 180 / np.pi) % 180

def non_maximal_suppression(g_mag, g_dir):
    m, n = g_mag.shape
    output = np.zeros((m, n), dtype=np.float32)
    angle = g_dir
    for i in range(1, m-1):
        for j in range(1, n-1):
            q, r = 255, 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q, r = g_mag[i, j + 1], g_mag[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q, r = g_mag[i + 1, j + 1], g_mag[i - 1, j - 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q, r = g_mag[i + 1, j], g_mag[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q, r = g_mag[i - 1, j + 1], g_mag[i + 1, j - 1]
            if (g_mag[i, j] >= q) and (g_mag[i, j] >= r):
                output[i, j] = g_mag[i, j]
            else:
                output[i, j] = 0
    return output

def double_threshold(img, low_threshold, high_threshold):
    strong = 255
    weak = 75
    output = np.zeros_like(img)
    strong_indices = np.where(img >= high_threshold)
    weak_indices = np.where((img >= low_threshold) & (img < high_threshold))
    output[strong_indices] = strong
    output[weak_indices] = weak
    return output

def edge_tracking_by_hysteresis(img):
    m, n = img.shape
    strong = 255
    weak = 75
    final_output = np.zeros((m, n), dtype=np.uint8)
    for i in range(1, m-1):
        for j in range(1, n-1):
            if img[i, j] == weak:
                if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong) or
                    (img[i, j + 1] == strong) or (img[i, j - 1] == strong) or
                    (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong) or
                    (img[i - 1, j + 1] == strong) or (img[i + 1, j - 1] == strong)):
                    final_output[i, j] = strong
                else:
                    final_output[i, j] = 0
            elif img[i, j] == strong:
                final_output[i, j] = strong
    return final_output

nms_output = non_maximal_suppression(g_mag, g_dir)
thresholded_output = double_threshold(nms_output, 10, 250)
final_edges = edge_tracking_by_hysteresis(thresholded_output)

images = (image, nms_output, thresholded_output, final_edges)

cv2.imshow('all', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()
