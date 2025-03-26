# Implement HOG Descriptor.

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)

CELL_SIZE = 8
BINS = 9
STRIDE = 2
BLOCK_SIZE = 2

gx_kernel = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])

gy_kernel = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])

gx = cv2.filter2D(image, ddepth=-1, kernel=gx_kernel)
gy = cv2.filter2D(image, ddepth=-1, kernel=gy_kernel)

g_mag = np.sqrt(gx**2 + gy**2)
g_dir = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180


def compute_histogram(mag, dir):
    hist = np.zeros(BINS)
    bin_width = 180 / BINS
    for y in range(mag.shape[0]):
        for x in range(mag.shape[1]):
            angle = dir[y, x]
            bin = int(angle // bin_width)
            weight = mag[y, x]
            next_bin = (bin + 1) % BINS
            bin_ratio = (angle % bin_width) / bin_width
            hist[bin] += weight * (1 - bin_ratio)
            hist[next_bin] += weight * bin_ratio
    return hist


h, w = image.shape
cells_y = h // CELL_SIZE
cells_x = w // CELL_SIZE
cell_histograms = np.zeros((cells_y, cells_x, BINS))

for y in range(cells_y):
    for x in range(cells_x):
        cell_mag = g_mag[y*CELL_SIZE:(y+1)*CELL_SIZE, x*CELL_SIZE:(x+1)*CELL_SIZE]
        cell_dir = g_dir[y*CELL_SIZE:(y+1)*CELL_SIZE, x*CELL_SIZE:(x+1)*CELL_SIZE]
        cell_histograms[y, x] = compute_histogram(cell_mag, cell_dir)

blocks_y = cells_y - BLOCK_SIZE + 1
blocks_x = cells_x - BLOCK_SIZE + 1
hog_features = []

for y in range(blocks_y):
    for x in range(blocks_x):
        block = cell_histograms[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE].flatten()
        norm = np.sqrt(np.sum(block**2) + 1e-6)
        hog_features.extend(block / norm)

def sliding_window(image, window_size, step_size):
    h, w = image.shape
    for y in range(0, h - window_size[1] + 1, step_size):
        for x in range(0, w - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

window_size = (64, 128)
for (x, y, window) in sliding_window(image, window_size, STRIDE):
    if window.shape[0] == window_size[1] and window.shape[1] == window_size[0]:
        features = np.array(hog_features)
        print("Window position:", (x, y), "Feature length:", len(features))