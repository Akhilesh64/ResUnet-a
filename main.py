import logging
import os, numpy as np
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
import tensorflow as tf
from model import ResUnet
from loss import Tanimoto_dual_loss
from tf.keras.optimizers import *
from batch_preprocess import DataParser

loss = Tanimoto_dual_loss()

losses = {'segmentation': loss, 'boundary': loss, 'distance': loss}

label_dict = {'(0, 0, 0)' : 0, '(255, 255, 255)' : 1}

parser = argparse.ArgumentParser()
parser.add_argument("--image_size",
                    help="Input image size for model.", type=int, default=256, choices=[256,448])
parser.add_argument("--batch_size",
                    help="Batch size for model.", type=int, default=8)
parser.add_argument("--num_classes",
                    help="Number of classes for the model.", type=int, default=2)
parser.add_argument("--validation_split",
                    help="Number of classes for the model.", type=float, default=0.1)
parser.add_argument("--epochs",
                    help="Number of epochs for the model.", type=int, default=200)
parser.add_argument("--image_path",
                    help="Training images directory path.", type=str, required=True)
parser.add_argument("--gt_path",
                    help="Training groundtruth directory path.", type = str, required=True)
parser.add_argument("--layer_norm",
                    help="Type of normalization for conv layers.", type = str, choices=['batch','instance'], default = 'batch')
parser.add_argument("--model_save_path",
                    help="Path to save model.", type = str, required=True)
parser.add_argument("--checkpoint_mode",
                    help="Mode of checkpointing.", type = str, default='epochs', choices=['epochs','best'])
args = parser.parse_args()

if not os.path.exists(args.model_save_path):
    os.makedirs(args.model_save_path)

if args.checkpoint_mode == 'best':
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.model_save_path, 'model_best.h5'),
        monitor='val_loss',
        mode='auto', verbose = 1,
        save_best_only=True)

if args.checkpoint_mode == 'epochs':
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.model_save_path, 'model.h5'),
        monitor='val_loss',
        mode='auto', verbose = 1,
        save_best_only=False, save_freq='epoch', period=10)
   

if args.image_size == 256:
    print('Training ResUnet-a d7v2 model !!!')
elif args.image_size == 448:
    print('Training ResUnet-a d6 model !!!')

resunet_a = ResUnet(args.num_classes, (args.image_size, args.image_size, 3), args.layer_norm)
model = resunet_a.build_model()
# model.summary()

metrics_dict = {'segmentation': ['accuracy']}
              
model.compile(optimizer=Adam(), loss=losses, metrics=metrics_dict)

metrics_names = ['loss', 'seg_loss', 'bound_loss', 'dist_loss', 'seg_accuracy']

dataParser = DataParser(args.image_path, args.gt_path, label_dict, args.validation_split, args.batch_size, args.image_size, args.num_classes)

def generate_minibatches(dataParser, train=True):
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.training_ids, dataParser.batch_size, replace=False)
        else:
            batch_ids = np.random.choice(dataParser.validation_ids, dataParser.batch_size, replace=False)
        images, labels = dataParser.get_batch(batch_ids)
        yield(images, labels)

model.fit_generator(generate_minibatches(dataParser),
                        steps_per_epoch=dataParser.steps_per_epoch,
                        epochs=args.epochs,
                        validation_data=generate_minibatches(dataParser, train=False),
                        validation_steps=dataParser.validation_steps,
                        callbacks=[checkpoint])
