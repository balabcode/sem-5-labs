import cv2
import numpy as np

image = cv2.imread('/home/Student/Downloads/tokyo.jpeg')
kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

filter1_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

cv2.imshow('Kernel 1', np.hstack((image, filter1_img)))
cv2.waitKey(0)
cv2.imwrite('/home/Student/Downloads/tokyo_filter1.jpeg', filter1_img)

kernel2 = np.ones((5,5), np.float32) / 25
filter2_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
cv2.imshow('Kernel 2', np.hstack((image, filter2_img)))
cv2.waitKey(0)
cv2.imwrite('/home/Student/Downloads/tokyo_filter2.jpeg', filter2_img)

cv2.destroyAllWindows()
