# Implement an image stitching algorithm using feature matching. Extract features from overlapping
# images, match the descriptors, estimate the transformation, and stitch the images together to create a
# seamless panoramic image.

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

def stitch_images(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H = estimate_homography(src, dst)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    c1 = [[0, 0], [0, h1], [w1, 0], [w1, h1]].astype(np.float32).reshape(-1, 1, 2)
    c2 = [[0, 0], [0, h2], [w2, 0], [w2, h2]].astype(np.float32).reshape(-1, 1, 2)
    c2_transformed = cv2.perspectiveTransform(c2, H)

    all_c = np.concatenate((c1, c2_transformed), axis=0)
    [x_min, y_min] = all_c.min(axis=0).astype(np.int32).flatten()
    [x_max, y_max] = all_c.max(axis=0).astype(np.int32).flatten()

    trans_dist = [-x_min, -y_min]
    H_trans = np.array([[1, 0, trans_dist[0]], [0, 1, trans_dist[1]], [0, 0, 1]])

    stitched_img = cv2.warpPerspective(img2, H_trans @ H, (x_max - x_min, y_max - y_min))
    stitched_img[trans_dist[1]:h1 + trans_dist[1], trans_dist[0]:w1 + trans_dist[0]] = img1

    return stitched_img

img1 = cv2.imread('../assets/sift/box.png')
img2 = cv2.imread('../assets/sift/box_in_scene.png')

stitched_result = stitch_images(img1, img2)
show(stitched_result)
