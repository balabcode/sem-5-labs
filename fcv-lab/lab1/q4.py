import cv2
import numpy as np
# Write a simple program to draw rectangle.

img = np.zeros((512,512,3), np.uint8)
img = cv2.rectangle(img,(10,10),(502, 502),(0,0,255),3)

cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
