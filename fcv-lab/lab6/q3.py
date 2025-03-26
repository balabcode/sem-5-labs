# Implement the homography estimation algorithm using feature matches. Given a set of matched
# keypoints, use techniques like RANSAC to estimate the homography matrix that represents the
# geometric transformation between two images.

import cv2
import numpy as np

def show(image, resized_height=360):
    percent = resized_height / len(image)
    resized_width = int(percent * len(image[0]))
    gray = cv2.resize(image, (resized_width, resized_height))
    cv2.imshow('cringe', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def estimate_homography(src, dst):
    A = []
    for i in range(len(src)):
        x, y = src[i][0]
        u, v = dst[i][0]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    return H / H[2, 2]

reference = cv2.imread('../assets/sift/box.png', 0)
scene = cv2.imread('../assets/sift/box_in_scene.png', 0)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(reference, None)
kp2, des2 = sift.detectAndCompute(scene, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

src = [kp1[m.queryIdx].pt for m in good].astype(np.float32).reshape(-1, 1, 2)
dst = [kp2[m.trainIdx].pt for m in good].astype(np.float32).reshape(-1, 1, 2)

H = estimate_homography(src, dst)

matchesMask = [1] * len(good)
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
results = cv2.drawMatchesKnn(reference, kp1, scene, kp2, good, None, **draw_params)

show(results)

