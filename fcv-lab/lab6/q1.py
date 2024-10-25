# Implement the nearest neighbour matching algorithm for feature descriptors. You may use any of the
# descriptor implemented earlier. Given two sets of descriptors, calculate the pairwise distances and find
# the best match for each descriptor.

import cv2
import numpy as np

def compute_pairwise_distances(set1, set2):
    return np.linalg.norm(set1[:, np.newaxis] - set2[np.newaxis, :], axis=2)

def nearest_neighbors(set1, set2, k=1):
    distances = compute_pairwise_distances(set1, set2)
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)
    return nearest_indices, nearest_distances

img1 = cv2.imread('../assets/common/image1.png', 0)
img2 = cv2.imread('../assets/common/image2.png', 0)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

if des1 is not None and des2 is not None:
    matches_indices, matches_distances = nearest_neighbors(des1, des2, k=1)
    for i, (match_idx, distance) in enumerate(zip(matches_indices, matches_distances)):
        print(f"Descriptor {i} matches descriptor {match_idx[0]} with distance {distance[0]:.2f}")

    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, 
                                    [cv2.DMatch(i, match_idx[0], distance[0]) for i, match_idx, distance in zip(range(len(matches_indices)), matches_indices, matches_distances)], 
                                    None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow('Matched Keypoints', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
