import cv2

# Write a simple program to Resizing the Image.

img = cv2.imread('q1_img.png')

resized_img = cv2.resize(img, (500, 500))

cv2.imshow('Resized', resized_img)
cv2.waitKey(0)

cv2.imwrite('q5_resized.png', resized_img)
cv2.destroyAllWindows()
