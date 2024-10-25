# Implement the ratio test for feature matching. Given a set of descriptors and their nearest neighbors,
# apply the ratio test to filter out unreliable matches and keep only the robust matches.

import cv2
import numpy as np


def show(image, resized_height=360):
    percent = resized_height / len(image)
    resized_width = int(percent * len(image[0]))
    gray = cv2.resize(image,(resized_width,resized_height))


    cv2.imshow('cringe', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 
reference = cv2.imread('../assets/sift/box.png', 0)
scene = cv2.imread('../assets/sift/box_in_scene.png', 0)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(reference, None)
kp2, des2 = sift.detectAndCompute(scene, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

results = cv2.drawMatchesKnn(reference,kp1,scene,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

show(results)