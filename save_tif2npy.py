import numpy as np
from osgeo import gdal

def load_tiff_image(patch):
    # Read tiff Image
    print(patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    return img

img_train = load_tiff_image('/content/drive/MyDrive/top_potsdam_2_10_RGB.tif')
print(img_train.shape)
np.save('Image_Train.npy', img_train)
del img_train

ref_train = load_tiff_image('/content/drive/MyDrive/top_potsdam_2_10_label.tif')
print(ref_train.shape)
np.save('Label_Train.npy', ref_train)
del ref_train

