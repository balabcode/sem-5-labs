import cv2

# Write a simple program to draw rectangle.

img = cv2.imread('q1_img.png')

img = cv2.rectangle(img,(0,0),(800,800),(0,0,255),3)

cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
