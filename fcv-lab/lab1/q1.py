import cv2

# Write a simple program to read, display, and write an image.

img = cv2.imread('q1_img.png', 0)
cv2.imshow('Nice Pixel Art :)', img)
cv2.waitKey(0)

cv2.imwrite('q1_bw_img.png', img)

cv2.destroyAllWindows()
