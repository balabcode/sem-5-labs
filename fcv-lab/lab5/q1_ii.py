# FAST Corner Detection

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)
h, w = image.shape

R = 3
n = 12
THRESH = 30

corners = np.zeros((h, w), dtype=np.uint8)

for y in range(R, h - R):
    for x in range(R, w - R):
        p = float(image[y, x])

        circle = [
            (x + int(R * np.cos(2 * np.pi * i / n)),
             y - int(R * np.sin(2 * np.pi * i / n)))
            for i in range(n)
        ]

        above_threshold = 0
        for px, py in circle:
            if 0 <= px < w and 0 <= py < h:
                if abs(float(image[py, px]) - p) > THRESH:
                    above_threshold += 1

        if above_threshold > 9:
            corners[y, x] = 255

images = (image, corners)
cv2.imshow('images', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()
