#Implementation of SIFT descriptor using Opencv library

import cv2
from cv2 import SIFT_create
import numpy as np

# read images
img = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)
img = cv2.resize(img, (150,150))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# create sift object
sift = SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
keypoints_with_size = np.copy(img)
cv2.drawKeypoints(img, keypoints, keypoints_with_size, color = (255, 0, 0),
flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('all',keypoints_with_size)
cv2.waitKey(0)
cv2.destroyAllWindows()
