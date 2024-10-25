# Extend the solved exercise to estimate the intrinsic parameter and the extrinsic parameter.

import numpy as np
import cv2
import glob

CHECKERBOARD_SIZE = (12, 12)
object_points, image_points = [], []
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
image_files = glob.glob('../assets/calib_example/*.tif')

for filename in image_files:
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_image, CHECKERBOARD_SIZE, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        object_points.append(objp)
        corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image_points.append(corners2)

object_points, image_points = np.array(object_points), np.array([pts.reshape(-1, 2) for pts in image_points])

def compute_homography(object_pts, img_pts):
    A = []
    for i in range(len(object_pts)):
        X, Y = object_pts[i, 0], object_pts[i, 1]
        x, y = img_pts[i, 0], img_pts[i, 1]
        A.append([-X, -Y, -1, 0, 0, 0, x * X, x * Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 3) / Vt[-1, -1]

homographies = [compute_homography(obj, img) for obj, img in zip(object_points, image_points)]

def compute_intrinsic(homographies):
    V = []
    def v_ij(h, i, j):
        return np.array([h[0, i] * h[0, j], h[0, i] * h[1, j] + h[1, i] * h[0, j], h[1, i] * h[1, j], 
                         h[2, i] * h[0, j] + h[0, i] * h[2, j], h[2, i] * h[1, j] + h[1, i] * h[2, j], h[2, i] * h[2, j]])
    
    for H in homographies:
        V.append(v_ij(H, 0, 1))
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
    _, _, Vt = np.linalg.svd(np.array(V))
    b = Vt[-1]
    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
    lambda_ = B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt(lambda_ / B[0, 0])
    beta = np.sqrt(lambda_ * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
    gamma = -B[0, 1] * alpha ** 2 * beta / lambda_
    u0 = gamma * v0 / alpha - B[0, 2] * alpha ** 2 / lambda_
    
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])

K = compute_intrinsic(homographies)
print("Intrinsic Camera Matrix K:\n", K)

def compute_extrinsics(K, H):
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = lam * np.dot(K_inv, h1)
    r2 = lam * np.dot(K_inv, h2)
    t = lam * np.dot(K_inv, h3)
    return np.column_stack((r1, r2, np.cross(r1, r2))), t

for i, H in enumerate(homographies):
    R, t = compute_extrinsics(K, H)
    print(f"\nExtrinsics for Image {i+1}:\nRotation Matrix R:\n{R}\nTranslation Vector T:\n{t}\n")