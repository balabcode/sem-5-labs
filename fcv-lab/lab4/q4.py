# Implement K means clustering algorithm.

import cv2
import numpy as np

K = 3
ITER = 100

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)
pixels = image.reshape(-1, 3)
centroids = pixels[np.random.choice(pixels.shape[0], K, replace=False)]

for _ in range(ITER):
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    
    new_centroids = np.array([pixels[clusters == i].mean(axis=0) for i in range(K)])
    
    if np.all(centroids == new_centroids):
        break
    
    centroids = new_centroids

segmented_img = centroids[clusters].reshape(image.shape).astype(np.uint8)
images = (image, segmented_img)

cv2.imshow('all', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()