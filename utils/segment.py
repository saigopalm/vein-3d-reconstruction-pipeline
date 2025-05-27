"""
This module provides a function to segment both left and right camera inputs and converts the segmentation output into binary masks suitable for 
keypoint matching and 3D reconstruction.
"""

from ultralytics import YOLO
import torch
import numpy as np

model = YOLO('model/best_ncnn_model', task='segment')

def segment_vein(frame_l, frame_r):
    results_l = model(frame_l, imgsz=(416,416))
    results_r = model(frame_r, imgsz=(416,416))
    left_mask = binary_mask(results_l[0].masks.data) if results_l[0].masks else None
    right_mask = binary_mask(results_r[0].masks.data) if results_r[0].masks else None
    return left_mask, right_mask

def binary_mask(img):
    c_mask = torch.zeros((416,416),dtype=torch.int8)
    for i in range(img.shape[0]):
        current_mask = img[i] * (i+1)
        c_mask = torch.where(current_mask > 0 ,current_mask,c_mask)
    img1 = c_mask
    new_array = np.zeros(img1.shape, dtype=np.uint8)
    new_array[np.where(img1 > 0)] = 255
    return new_array