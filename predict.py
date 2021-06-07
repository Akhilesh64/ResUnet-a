from utils import load_npy_image
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from tensorflow.keras.models import load_model
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

import ast
from sklearn.preprocessing import StandardScaler

root_path = './'
patch_size = 256
num_classes = 6
output_path = './results/'
img_test_path = 'Image_Train.npy'
model_path = './drive/MyDrive/results/best_model.h5'

def pred_recostruction(patch_size, pred_labels, binary_img_test_ref, img_type=1):
    # Patches Reconstruction
    if img_type == 1:
        stride = patch_size

        height, width = binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride
        #print(num_patches_h, num_patches_w)

        new_shape = (height, width)
        img_reconstructed = np.zeros(new_shape)
        cont = 0
        # rows
        for h in range(num_patches_h):
            # columns
            for w in range(num_patches_w):
                img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride] = pred_labels[cont]
                cont += 1
        print('Reconstruction Done!')
    if img_type == 2:
        stride = patch_size

        height, width = binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride

        new_shape = (height, width, 3)
        img_reconstructed = np.zeros(new_shape)
        cont = 0
        # rows
        for h in range(num_patches_h):
            # columns
            for w in range(num_patches_w):
                img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride, :] = pred_labels[cont]
                cont += 1
        print('Reconstruction Done!')
    return img_reconstructed

def convert_preds2rgb(img_reconstructed, label_dict):
    reversed_label_dict = {value:key for (key, value) in label_dict.items()}
    print(reversed_label_dict)
    height, width = img_reconstructed.shape
    img_reconstructed_rgb = np.zeros((height, width, 3))
    for h in range(height):
        for w in range(width):
            pixel_class = img_reconstructed[h, w]
            img_reconstructed_rgb[h, w, :] = ast.literal_eval(reversed_label_dict[pixel_class])
    print('Conversion to RGB Done!')
    return img_reconstructed_rgb.astype(np.uint8)

def extract_patches_train(img_test_normalized, patch_size):
    # Extract training patches manual
    stride = patch_size

    height, width, channel = img_test_normalized.shape
    #print(height, width)

    num_patches_h = height // stride
    num_patches_w = width // stride
    #print(num_patches_h, num_patches_w)

    new_shape = (num_patches_h*num_patches_w, patch_size, patch_size, channel)
    new_img = np.zeros(new_shape)
    print(new_img.shape)
    cont = 0
    # rows
    for h in range(num_patches_h):
        # columns
        for w in range(num_patches_w):
            new_img[cont] = img_test_normalized[h*stride:(h+1)*stride, w*stride:(w+1)*stride]
            cont += 1
    #print(cont)


    return new_img

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

img_test = load_npy_image(os.path.join(root_path, img_test_path)).astype(np.float32)
img_test_normalized = normalize_rgb(img_test)
# Transform the image into W x H x C shape
img_test_normalized = img_test_normalized.transpose((1, 2, 0))

label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1,
              '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4, '(255, 0, 0)' : 5}

patches_test = extract_patches_train(img_test_normalized, patch_size)
model = load_model(model_path, compile=False)
preds = model.predict(patches_test, batch_size=1)

# seg_preds = preds['seg']
# seg_pred = np.argmax(seg_preds, axis=-1)
patches_pred = [preds['seg'], preds['bound'], preds['dist'], preds['color']]
# img_reconstructed = pred_recostruction(patch_size, seg_pred,
#                                        img_test_normalized, img_type=1)
# img_reconstructed_rgb = convert_preds2rgb(img_reconstructed,
#                                           label_dict)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# plt.imsave(os.path.join(output_path, 'pred_seg_reconstructed.jpeg'),
#            img_reconstructed_rgb)

for i in range(len(patches_test)):
    for n_class in range(num_classes):
        task_pred = patches_pred[1]
        plt.imshow(task_pred[i, :, :, n_class],cmap=cm.Greys_r)
        plt.savefig(os.path.join(output_path, 'polygon_'+str(i+1)+'_class'+str(n_class)+'.png'))
