# Log Transforms
import cv2
import numpy as np

img = cv2.imread('../assets/common/image.png')
img = cv2.resize(img, (270,270))
img = img.astype(np.float32)

c = 255/(np.log(1+np.max(img)))

log_transformed = c * np.log(1+img)

log_transformed = np.array(log_transformed, dtype=np.uint8)

cv2.imwrite('../assets/lab2/l2sample_log_transformed.png', log_transformed)
