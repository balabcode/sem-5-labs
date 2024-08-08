# Write a program to read an image and perform unsharp masking.

import cv2
import numpy as np

images = []
SHARP_FACTOR = 1.5

image = cv2.imread('/home/Student/Downloads/tokyo.jpeg')/255
images.append(image)

blur_kernel = np.ones((5, 5), np.float32)/25
blurred_img = cv2.filter2D(image, -1, blur_kernel)
images.append(blurred_img)

detail = abs(image - blurred_img)
images.append(detail)

sharpened_image = image + (SHARP_FACTOR * detail)
images.append(sharpened_image)

cv2.imshow('all', np.hstack(tuple(images)))
cv2.waitKey(0)
cv2.destroyAllWindows()
