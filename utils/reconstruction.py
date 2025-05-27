"""
This module handles:
- Mapping of corresponding 2D points from stereo masks using a homography matrix.
- Triangulation of 3D points using a vectorized implementation of the Direct Linear Transform (DLT) method.
"""

import numpy as np
import tensorflow as tf

def compute_projectionMatrix(mtx1, mtx2, R, T):
    RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = mtx1 @ RT1
    RT2 = np.hstack((R, T))
    P2 = mtx2 @ RT2
    return P1, P2

def signal_points(mask, h):
    y, x = np.where(mask == 255)
    points1 = np.array(list(zip(y, x)))
    ones = np.ones((points1.shape[0], 1))
    points1_h = np.concatenate((points1, ones), axis=1)
    points2 = (h @ points1_h.T).T
    points2 = points2[:, :2] / points2[:, 2:]
    return points1[:, [1,0]], points2[:, [1,0]]

def DLT_vectorized(P1, P2, points1, points2):
    A = tf.stack([
        tf.experimental.numpy.outer(points1[:, 1],P1[2, :]) - P1[1, :],
        P1[0, :] - tf.experimental.numpy.outer(points1[:, 0], P1[2, :]),
        tf.experimental.numpy.outer(points2[:, 1], P2[2, :]) - P2[1, :],
        P2[0, :] - tf.experimental.numpy.outer(points2[:, 0],P2[2, :])
    ])
    A = tf.transpose(A, (1,0,2))
    U, s, Vh = tf.linalg.svd(A, full_matrices=False)
    t = tf.reshape(Vh[:, 3, 3], (1, -1, 1))
    p3ds = Vh[:, 0:3, 3] / t
    return tf.reshape(p3ds, (-1, 3))