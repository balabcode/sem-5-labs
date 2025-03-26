import cv2

# Write a simple program to read, display, and write an image.

img = cv2.imread('../assets/common/image.png', 0)
cv2.imshow('Nice Pixel Art :)', img)
cv2.waitKey(0)

cv2.imwrite('../assets/lab1/l1q1_b_w.png', img)

cv2.destroyAllWindows()
