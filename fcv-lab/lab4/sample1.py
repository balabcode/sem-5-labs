import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)

ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(image, 120, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(image, 120, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(image, 120, 255, cv2.THRESH_TOZERO_INV)

images = (image, thresh1, thresh2, thresh3, thresh4, thresh5)

cv2.imshow('all', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()