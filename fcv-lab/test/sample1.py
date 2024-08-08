import cv2

image = cv2.imread('/home/Student/Downloads/tokyo.jpeg')
cv2.imshow('image', image)
cv2.waitKey(0)

# Gaussian Blur
gaussian = cv2.GaussianBlur(image, (7,7), 0)
cv2.imshow('Gaussian Blur', gaussian)
cv2.waitKey(0)

# Median Blur
median = cv2.medianBlur(image, 5)
cv2.imshow('Median Blur', median)
cv2.waitKey(0)

# Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blur', bilateral)
cv2.waitKey(0)

cv2.destroyAllWindows()
