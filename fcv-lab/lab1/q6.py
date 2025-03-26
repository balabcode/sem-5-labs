import cv2

# Write a simple program to Rotating the Image.

img = cv2.imread('../assets/common/image.png')

rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('rotated_img', rotated_img)
cv2.waitKey(0)

cv2.imwrite('../assets/lab1/l1q6_rotated.png', rotated_img)
cv2.destroyAllWindows()
