import cv2

# Write a simple program to Extracting the RGB values of a pixel.

img = cv2.imread('../assets/common/image.png')

# Color values of the 300x300 location in the image:
print(img[300,300])
cv2.destroyAllWindows()
