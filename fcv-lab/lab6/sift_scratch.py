import cv2
import numpy as np

image = cv2.imread("../assets/common/image_resized.png", 0)
print(image)

def find_gradients(image):
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    
    gx = cv2.filter2D(image, -1, kx)
    gy = cv2.filter2D(image, -1, ky)
    return gx, gy


# creating DoG pyramid
LEVELS = 5
gaussian_pyramid = []
temp_image = image.copy()
for i in range(1, LEVELS):
    temp_image = cv2.GaussianBlur(temp_image, (5,5), 0)
    temp_image = cv2.resize(temp_image, (temp_image.shape[1]//2 , temp_image.shape[0]//2))
    gaussian_pyramid.append(temp_image)
dog_pyramid = []
for i in range(1, LEVELS):
    dog_pyramid.append(cv2.subtract(gaussian_pyramid[i - 1], gaussian_pyramid[i]))

# computing keypoints
THRESH = 0.03
keypoints = []
for i, dog in enumerate(dog_pyramid):
    for y in range(1, dog.shape[0] - 1):
        for x in range(1, dog.shape[1] - 1):
            patch = dog[y - 1:y + 2, x - 1:x + 2]
            if (np.abs(dog[y, x]) > THRESH and (np.all(dog[y, x] >= patch) or np.all(dog[y, x] <= patch))):
                keypoints.append((x, y, i))

# computing orientations
orientations = []
for x, y, level in keypoints:
    gx, gy = find_gradients(image)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy[y, x], gx[y, x]) * (180 / np.pi) % 360
    orientations.append((x, y, level, direction))

# creating descriptors for every point
descriptors = []
for x, y, level, orientation in orientations:
    patch = gaussian_pyramid[level][y - 8:y + 8, x - 8:x + 8]
    patch = cv2.resize(patch, (16, 16)).flatten()
    descriptors.append(patch)


descriptors = np.array(descriptors)
print(descriptors)