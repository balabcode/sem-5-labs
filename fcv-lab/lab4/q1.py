# Write a program to create binary images using thresholding methods.

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)

threshold_image = np.where(image > 127, 191, 63).astype(np.uint8)

images = (image, threshold_image)
cv2.imshow('all', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()