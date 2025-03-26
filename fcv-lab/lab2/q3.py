# Write a program to Resizing the Image and cropping an image.

import cv2

img = cv2.imread('../assets/common/image.png')
img = cv2.resize(img, (270, 270))

cv2.imshow('image', img[:200][:200])
cv2.waitKey(0)
cv2.destroyAllWindows()
