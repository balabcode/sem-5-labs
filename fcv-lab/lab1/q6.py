import cv2

# Write a simple program to Rotating the Image.

img = cv2.imread('q1_img.png')

rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('rotated_img', rotated_img)
cv2.waitKey(0)

cv2.imwrite('q6_rotated_img.png', rotated_img)
cv2.destroyAllWindows()
