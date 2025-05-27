import cv2
import numpy as np
from config.camera_config import setup_camera
from utils.segment import segment_vein
from utils.keypoints import keypoint_match, homography
from utils.reconstruction import compute_projectionMatrix, signal_points, DLT_vectorized
from utils.visualization import plot_3d

# loading camera parameters
mtx1 = np.load("calibration/camera_params/mtx1.npy")
mtx2 = np.load("calibration/camera_params/mtx2.npy")
dist1 = np.load("calibration/camera_params/dist1.npy")
dist2 = np.load("calibration/camera_params/dist2.npy")
R = np.load("calibration/camera_params/R.npy")
T = np.load("calibration/camera_params/T.npy")

# loading camera matrices
newcameramtx1, _ = cv2.getOptimalNewCameraMatrix(mtx1, dist1, (416, 416), 1, (416, 416))
newcameramtx2, _ = cv2.getOptimalNewCameraMatrix(mtx2, dist2, (416, 416), 1, (416, 416))

picam1 = setup_camera(0)
picam2 = setup_camera(1)

while True:
    frame_l = picam1.capture_array()
    frame_r = picam2.capture_array()

    left_mask, right_mask  = segment_vein(frame_l, frame_r)

    if left_mask is not None and right_mask is not None:
        left_mask = cv2.undistort(left_mask, mtx1, dist1, None, newcameramtx1)
        right_mask = cv2.undistort(right_mask, mtx2, dist2, None, newcameramtx2)

        matches, match_coordinates = keypoint_match(left_mask, right_mask)
        print(matches)

        if matches > 4:
            h = homography(match_coordinates)
            if h is not None:
                P1, P2 = compute_projectionMatrix(mtx1, mtx2, R, T)
                points1, points2 = signal_points(left_mask, h)
                p3ds = DLT_vectorized(P1, P2, points1, points2)
                plot_3d(p3ds.numpy())
            else:
                print("No homography found")
        else:
            print("Not enough matches found")
    else:
        print("No mask found")
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
