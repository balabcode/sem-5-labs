# Write a program to obtain gradient of an image.

import cv2
import numpy as np

image = cv2.imread('/home/Student/Downloads/tokyo.jpeg')

Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

Gy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])


gradient_x_mat = np.dot(image, Gx)
x_gradient = sum(gradient_x_mat)

print(x_gradient)