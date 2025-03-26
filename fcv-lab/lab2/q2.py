# Write a program to read an input image, reference image, and perform histogram specification.

import cv2
import numpy as np

img = cv2.imread('../assets/common/image.png', 0)
img2 = cv2.imread('../assets/common/image2.png', 0)
img2 = cv2.resize(img2, (270, 270))
img = cv2.resize(img, (270,270))

data = img.flatten()
histogram, _ = np.histogram(data, bins=256, range=(0, 256))
histogram = histogram/(270*270)

cdf = np.cumsum(histogram)

new_img = []
for num in img2.flatten():
    new_img.append(round(cdf[num]*255))

new_img = np.array(new_img).reshape(270,270).astype(np.uint8)
cv2.imshow('image', new_img)
cv2.imwrite('../assets/lab2/q2_specification.png', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
