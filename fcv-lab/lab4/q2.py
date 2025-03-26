# Write a program to detect lines using Hough transform.

import cv2
import numpy as np

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", 0)
edges = cv2.Canny(image, 50, 150)

h,w = edges.shape
diag = int(np.sqrt(h**2 + w**2))
acc = np.zeros((2*diag, 180), dtype=np.uint16)

THRESH = 50

for y in range(h):
    for x in range(w):
        if edges[y,x] > 0:
            for angle in range(180):
                theta = np.deg2rad(angle)
                rho = int(x*np.cos(theta) + y*np.sin(theta)) + diag
                acc[rho,angle] += 1

lines = np.argwhere(acc > THRESH)

image_with_lines = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png")
for rho, angle in lines:
    theta = np.deg2rad(angle)
    a = np.cos(theta)
    b = np.sin(theta)
    
    x0 = a * (rho-diag)
    y0 = b * (rho-diag)
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(x0 - 1000*(a))
    
    cv2.line(image_with_lines, (x1,y1), (x2,y2), (0,255,0), 1)

cv2.imshow('image', image_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()