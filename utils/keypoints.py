"""
This module uses ORB (Oriented FAST and Rotated BRIEF) to detect and match keypoints between two stereo image and filters good matches based on the 
vertical alignment of keypoints. The matched coordinates are then used to compute a homography matrix that can warp one image onto the perspective of the other.
"""
import cv2
import numpy as np

def keypoint_match(img1, img2):
    orb = cv2.ORB_create(nfeatures=1500, edgeThreshold=40, patchSize=40, WTA_K=4)
    KeyPoints1, des1 = orb.detectAndCompute(img1, None)
    KeyPoints2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matchCoordinatesL, matchCoordinatesR = [], []
    for match in matches:
        matchCoordinatesL.append([KeyPoints1[match.queryIdx].pt])
        matchCoordinatesR.append([KeyPoints2[match.trainIdx].pt])
    good_matches, L2, R2 = [], [], []
    for i in range(len(matchCoordinatesL)):
        x, y = matchCoordinatesL[i], matchCoordinatesR[i]
        if 2 <= abs(x[0][1] - y[0][1]) <= 10:
            L2.append(x)
            R2.append(y)
            good_matches.append(matches[i])
    matchCoordinatesL = np.array(L2).reshape(len(L2), 2)
    matchCoordinatesR = np.array(R2).reshape(len(R2), 2)
    matchCoordinates = np.vstack((matchCoordinatesL, matchCoordinatesR)).reshape(2, -1, 2)
    return len(good_matches), matchCoordinates

def homography(matchCoordinates):
    h, status = cv2.findHomography(matchCoordinates[0], matchCoordinates[1], cv2.RANSAC, 100)
    return h