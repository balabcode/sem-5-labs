# Gamma Transforms
import cv2
import numpy as np

img = cv2.imread('../assets/common/image.png')
img = cv2.resize(img,(270,270))

a = []
for gamma in [0.1, 0.5, 1.2, 2.2]:
    gamma_corrected = np.array(255*(img/255)**gamma, dtype=np.uint8)
    a.append(gamma_corrected)

a = np.hstack(tuple(a)).astype(np.uint8)
cv2.imshow('a', a)
cv2.imwrite('../assets/lab2/s2_gamma_corrected.png', a)
cv2.waitKey(0)
cv2.destroyAllWindows()
