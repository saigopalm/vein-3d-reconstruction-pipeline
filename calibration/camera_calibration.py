'''
This code calibrates the both the cameras and saves the parameters in calibration/camera_params
'''

import cv2 as cv
import numpy as np
import glob
import os

# configuration
rows, columns = 6, 7
square_size = 4.235
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
conv_size = (11, 11)

left_imgs = sorted(glob.glob('left/*.png'))
right_imgs = sorted(glob.glob('right/*.png'))
output_dir_left = "out_left/"
output_dir_right = "out_right/"
os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

# Calibration object points
objp = np.zeros((rows * columns, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
objp *= square_size


def calibrate_camera(images_left, images_right):
    objpoints, imgpoints_left, imgpoints_right = [], [], []

    for i, (img_l, img_r) in enumerate(zip(images_left, images_right), 1):
        gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

        ret_l, corners_l = cv.findChessboardCorners(gray_l, (rows, columns), 
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        ret_r, corners_r = cv.findChessboardCorners(gray_r, (rows, columns), 
            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if ret_l and ret_r:
            corners_l = cv.cornerSubPix(gray_l, corners_l, conv_size, (-1, -1), criteria)
            corners_r = cv.cornerSubPix(gray_r, corners_r, conv_size, (-1, -1), criteria)

            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)
            objpoints.append(objp)

            cv.imwrite(os.path.join(output_dir_left, f"{i}.png"), 
                       cv.drawChessboardCorners(img_l.copy(), (rows, columns), corners_l, ret_l))
            cv.imwrite(os.path.join(output_dir_right, f"{i}.png"), 
                       cv.drawChessboardCorners(img_r.copy(), (rows, columns), corners_r, ret_r))

    def calibrate_single(imgpoints, image_shape):
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
        error = np.mean([cv.norm(ip, cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)[0], cv.NORM_L2) 
                         / len(ip) for i, ip in enumerate(imgpoints)])
        return mtx, dist, error

    image_shape = (images_left[0].shape[1], images_left[0].shape[0])
    mtx1, dist1, error1 = calibrate_single(imgpoints_left, image_shape)
    mtx2, dist2, error2 = calibrate_single(imgpoints_right, image_shape)

    print(f"Reprojection Error (Left): {error1:.4f}")
    print(f"Reprojection Error (Right): {error2:.4f}")
    return mtx1, dist1, mtx2, dist2


def stereo_calibrate(mtx1, dist1, mtx2, dist2, images_left, images_right):
    objpoints, imgpoints_left, imgpoints_right = [], [], []

    for img_l, img_r in zip(images_left, images_right):
        gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

        ret_l, corners_l = cv.findChessboardCorners(gray_l, (rows, columns), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        ret_r, corners_r = cv.findChessboardCorners(gray_r, (rows, columns), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if ret_l and ret_r:
            corners_l = cv.cornerSubPix(gray_l, corners_l, conv_size, (-1, -1), criteria)
            corners_r = cv.cornerSubPix(gray_r, corners_r, conv_size, (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)

    shape = (images_left[0].shape[1], images_left[0].shape[0])
    flags = cv.CALIB_FIX_INTRINSIC

    ret, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx1, dist1, mtx2, dist2, shape, criteria=criteria, flags=flags
    )

    print(f"Stereo Calibration RMS Error: {ret:.4f}")
    return R, T


# Run calibration
images_left = [cv.imread(f) for f in left_imgs]
images_right = [cv.imread(f) for f in right_imgs]
mtx1, dist1, mtx2, dist2 = calibrate_camera(images_left, images_right)
np.save("camera_params/mtx1.npy", mtx1)
np.save("camera_params/dist1.npy", dist1)
np.save("camera_params/mtx2.npy", mtx2)
np.save("camera_params/dist2.npy", dist2)

R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, images_left, images_right)
np.save("camera_params/R.npy", R)
np.save("camera_params/T.npy", T)