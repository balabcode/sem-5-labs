# Write a program to detect edges in a image

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)

kernel =    np.array([[0,1,0],
                       [1,4,1],
                       [0,1,0]])

edges = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

images = (image, edges)

cv2.imshow('all', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()