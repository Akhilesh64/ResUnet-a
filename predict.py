import cv2, os
import numpy as np
from tensorflow.keras.models import load_model
from utils import normalize_rgb
from tqdm import tqdm
import argparse
import tensorflow_addons as tfa

label_dict = {'(0, 0, 0)':0, '(255, 255, 255)':1}

parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=256, choices=[256,448])
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required = True)
args = parser.parse_args()

if not os.path.exists(args.output_path):
      os.makedirs(args.output_path)

model = load_model(args.model_path, compile=False, custom_objects={'InstanceNormalization':tfa.layers.InstanceNormalization()})

for file in tqdm(os.listdir(args.image_path)):
    name, ext = os.path.splitext(file)
    im = cv2.imread(os.path.join(args.image_path,file))
    im = cv2.resize(im, (args.image_size, args.image_size))
    im = normalize_rgb(im)
    im = np.expand_dims(im, axis=0)

    pred = model.predict(im, batch_size = 1)
    # print(pred.shape)
    pred = pred[0]
    pred = pred[0, :, :, 0]*255
    
    pred = cv2.threshold(pred,np.mean(pred)+1.2*np.std(pred),255,cv2.THRESH_BINARY)[1]
    pred = cv2.resize(pred,(480,464))
    cv2.imwrite(os.path.join(args.output_path, name+'.png'),pred)
