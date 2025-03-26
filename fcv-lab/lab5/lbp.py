# Implement LBP Descriptor

import cv2
import numpy as np

def compute_lbp(image, neighborhood_size=3):
    h, w = image.shape
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    
    radius = (neighborhood_size - 1) // 2
    P = neighborhood_size ** 2 - 1
    
    offsets = [(int(radius * np.sin(2 * np.pi * p / P)),
                int(radius * np.cos(2 * np.pi * p / P)))
               for p in range(P)]

    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            center_pixel = image[y, x]
            binary_code = 0
            
            for i, (dy, dx) in enumerate(offsets):
                neighbor_y, neighbor_x = y + dy, x + dx
                if image[neighbor_y, neighbor_x] >= center_pixel:
                    binary_code |= (1 << i)
            
            lbp_image[y, x] = binary_code
    
    return lbp_image

image = cv2.imread(r"C:\Users\bala\Documents\code\FCV-Lab-main\assets\common\image_resized.png", cv2.IMREAD_GRAYSCALE)

lbp_image = compute_lbp(image, neighborhood_size=3)

images = (image, lbp_image)
cv2.imshow('all', np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()