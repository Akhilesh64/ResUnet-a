from utils import np, load_npy_image, data_augmentation
import tensorflow as tf

from multitasking_utils import get_boundary_label, get_distance_label
import argparse
import os

from skimage.util.shape import view_as_windows

import gc
import psutil
import cv2
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_patches(image, reference, patch_size, stride):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image,
                                             window_shape_array, step=stride))

    patches_ref = np.array(view_as_windows(reference,
                                           window_shape_ref, step=stride))

    print('Patches extraidos')
    print(patches_array.shape)
    num_row, num_col, p, row, col, depth = patches_array.shape

    print('fazendo reshape')
    patches_array = patches_array.reshape(num_row*num_col, row, col, depth)
    print(patches_array.shape)
    patches_ref = patches_ref.reshape(num_row*num_col, row, col)
    print(patches_ref.shape)

    return patches_array, patches_ref


def binarize_matrix(img_train_ref, label_dict):
    # Create binarized matrix
    w = img_train_ref.shape[0]
    h = img_train_ref.shape[1]
    # c = img_train_ref.shape[2]
    # binary_img_train_ref = np.zeros((1,w,h))
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


def normalize_rgb(img, norm_type=1):
    # OBS: Images need to be converted to before float32 to be normalized
    # TODO: Maybe should implement normalization with StandardScaler
    # Normalize image between [0, 1]
    if norm_type == 1:
        img /= 255.
    # Normalize image between [-1, 1]
    elif norm_type == 2:
        img /= 127.5 - 1.
    elif norm_type == 3:
        image_reshaped = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
        scaler = StandardScaler()
        scaler = scaler.fit(image_reshaped)
        image_normalized = scaler.fit_transform(image_reshaped)
        img = image_normalized.reshape(img.shape[0], img.shape[1], img.shape[2])

    return img


def normalize_hsv(img, norm_type=1):
    # OBS: Images need to be converted to before float32 to be normalized
    # TODO: Maybe should implement normalization with StandardScaler
    # Normalize image between [0, 1]
    if norm_type == 1:
        img[:, :, 0] /= 179.
        img[:, :, 1] /= 255.
        img[:, :, 2] /= 255.
    # Normalize image between [-1, 1]
    elif norm_type == 2:
        img[:, :, 0] /= 89.5 - 1.
        img[:, :, 1] /= 127.5 - 1.
        img[:, :, 2] /= 127.5 - 1.
    elif norm_type == 3:
        image_reshaped = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
        scaler = StandardScaler()
        scaler = scaler.fit(image_reshaped)
        image_normalized = scaler.fit_transform(image_reshaped)
        img = image_normalized.reshape(img.shape[0], img.shape[1], img.shape[2])

    return img


parser = argparse.ArgumentParser()
parser.add_argument("--norm_type",
                    help="Choose type of normalization to be used", type=int,
                    default=1, choices=[1, 2, 3])
parser.add_argument("--patch_size",
                    help="Choose size of patches", type=int, default=256)
parser.add_argument("--stride",
                    help="Choose stride to be using on patches extraction",
                    type=int, default=32)
parser.add_argument("--num_classes",
                    help="Choose number of classes to convert \
                    labels to one hot encoding", type=int, default=5)
parser.add_argument("--data_aug",
                    help="Choose number of classes to convert \
                    labels to one hot encoding", type=str2bool, default=True)
args = parser.parse_args()

print('='*50)
print('Parameters')
print(f'patch size={args.patch_size}')
print(f'stride={args.stride}')
print(f'Number of classes={args.num_classes} ')
print('='*50)

root_path = './'
# Load images
img_train_path = 'Image_Train.npy'
img_train = load_npy_image(os.path.join(root_path,
                                        img_train_path))
# Convert shape from C x H x W --> H x W x C
img_train = img_train.transpose((1, 2, 0))
# img_train_normalized = normalization(img_train)
print(img_train.shape)

# Load reference
img_train_ref_path = 'Label_Train.npy'
img_train_ref = load_npy_image(os.path.join(root_path, img_train_ref_path))
# Convert from C x H x W --> H x W x C
img_train_ref = img_train_ref.transpose((1, 2, 0))
print(img_train_ref.shape)

label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1,
              '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4, '(255, 0, 0)':5}

binary_img_train_ref = binarize_matrix(img_train_ref, label_dict)
del img_train_ref

# stride = patch_size
patches_tr, patches_tr_ref = extract_patches(img_train,
                                             binary_img_train_ref,
                                             args.patch_size, args.stride)

process = psutil.Process(os.getpid())
print('[CHECKING MEMORY]')

print(process.memory_percent())
del binary_img_train_ref, img_train

print(process.memory_percent())
gc.collect()
print(process.memory_percent())

print('saving images...')
folder_path = f'./DATASETS/patch_size={args.patch_size}_' + \
            f'stride={args.stride}_norm_type={args.norm_type}_data_aug={args.data_aug}'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    os.makedirs(os.path.join(folder_path, 'train'))
    os.makedirs(os.path.join(folder_path, 'labels'))
    os.makedirs(os.path.join(folder_path, 'labels/seg'))
    os.makedirs(os.path.join(folder_path, 'labels/bound'))
    os.makedirs(os.path.join(folder_path, 'labels/dist'))
    os.makedirs(os.path.join(folder_path, 'labels/color'))


def filename(i):
    return f'patch_{i}.npy'


print(f'Number of patches: {len(patches_tr)}')
if args.data_aug:
    print(f'Number of patches expected: {len(patches_tr)*5}')
for i in tqdm(range(len(patches_tr))):
    if args.data_aug:
        img_aug, label_aug = data_augmentation(patches_tr[i], patches_tr_ref[i])
    else:
        img_aug, label_aug = np.expand_dims(patches_tr[i], axis=0), np.expand_dims(patches_tr_ref[i], axis=0)
    label_aug_h = tf.keras.utils.to_categorical(label_aug, args.num_classes)
    for j in range(len(img_aug)):
        # Input image RGB
        # Float32 its need to train the model
        img_float = img_aug[j].astype(np.float32)
        img_normalized = normalize_rgb(img_float, norm_type=args.norm_type)
        np.save(os.path.join(folder_path, 'train', filename(i*5 + j)),
                img_normalized)
        # All multitasking labels are saved in one-hot
        # Segmentation
        np.save(os.path.join(folder_path, 'labels/seg', filename(i*5 + j)),
                label_aug_h[j].astype(np.float32))
        # Boundary
        bound_label_h = get_boundary_label(label_aug_h[j]).astype(np.float32)
        np.save(os.path.join(folder_path, 'labels/bound', filename(i*5 + j)),
                bound_label_h)
        # Distance
        dist_label_h = get_distance_label(label_aug_h[j]).astype(np.float32)
        np.save(os.path.join(folder_path, 'labels/dist', filename(i*5 + j)),
                dist_label_h)
        # Color
        hsv_patch = cv2.cvtColor(img_aug[j],
                                 cv2.COLOR_RGB2HSV).astype(np.float32)
        # Float32 its need to train the model
        hsv_patch = normalize_hsv(hsv_patch, norm_type=args.norm_type)
        np.save(os.path.join(folder_path, 'labels/color', filename(i*5 + j)),
                hsv_patch)
