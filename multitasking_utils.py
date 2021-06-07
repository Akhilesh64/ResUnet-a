import tensorflow as tf
import cv2
import numpy as np


def get_boundary_label(label, kernel_size=(3, 3)):
    _, _, channel = label.shape
    bounds = np.empty_like(label, dtype=np.float32)
    for c in range(channel):
        tlabel = label.astype(np.uint8)
        # Apply filter per channel
        temp = cv2.Canny(tlabel[:, :, c], 0, 1)
        tlabel = cv2.dilate(temp,
                            cv2.getStructuringElement(
                                cv2.MORPH_CROSS,
                                kernel_size),
                            iterations=1)
        # Convert to be used on training (Need to be float32)
        tlabel = tlabel.astype(np.float32)
        # Normalize between [0, 1]
        tlabel /= 255.
        bounds[:, :, c] = tlabel
    return bounds


def get_distance_label(label):
    label = label.copy()
    dists = np.empty_like(label, dtype=np.float32)
    for channel in range(label.shape[2]):
        patch = label[:, :, channel].astype(np.uint8)
        dist = cv2.distanceTransform(patch, cv2.DIST_L2, 0)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dists[:, :, channel] = dist

    return dists

