import numpy as np
import cv2, os
from sklearn.preprocessing import StandardScaler

def binarize_matrix(img_train_ref, label_dict):
    # Create binarized matrix
    w = img_train_ref.shape[0]
    h = img_train_ref.shape[1]
    binary_img_train_ref = np.full((w, h), -1, dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            r = img_train_ref[i][j][0]
            g = img_train_ref[i][j][1]
            b = img_train_ref[i][j][2]
            rgb = (r, g, b)
            rgb_key = str(rgb)
            binary_img_train_ref[i][j] = label_dict[rgb_key]

    return binary_img_train_ref

def split_pair_names(img_path, label_path):
    img_paths = sorted(os.listdir(img_path))
    label_paths = sorted(os.listdir(label_path))
    filenames = [(os.path.join(img_path,file1), os.path.join(label_path,file2)) for file1,file2 in zip(img_paths, label_paths)]
    return filenames

def normalize_rgb(img):
    im = img
    img = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
    scaler = StandardScaler()
    scaler = scaler.fit(img)
    img = scaler.fit_transform(img)
    img = img.reshape(im.shape[0], im.shape[1], im.shape[2])
    return img

def data_augmentation(image, labels):
    aug_imgs = np.zeros((3, image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
    aug_lbs = np.zeros((3, image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(0, len(aug_imgs)):
        aug_imgs[0, :, :, :] = image
        aug_imgs[1, :, :, :] = np.rot90(image, 1)
        aug_imgs[2, :, :, :] = np.flip(image, 1)

    for i in range(0, len(aug_lbs)):
        aug_lbs[0, :, :] = labels
        aug_lbs[1, :, :] = np.rot90(labels, 1)
        aug_lbs[2, :, :] = np.flip(labels, 1)

    return aug_imgs, aug_lbs

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
