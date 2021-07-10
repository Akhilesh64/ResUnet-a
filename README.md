# ResUnet-a

This repository contains implementation of the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data" in TensorFlow for the detection of plot boundaries specfically.

## Usage

1. Clone this repo using :
```
git clone https://github.com/Akhilesh64/ResUnet-a
``` 
2. Install the requiremnts using :
```
pip install -r requirements.txt
```
3. To start model training run the main.py file with following arguments :
```
python main.py --image_size 256 --batch_size 8 --num_classes 2 --validation_split 0.2 --epochs 100 --image_path ./images --gt_path ./gt --layer_norm batch --model_save_path ./ --checkpoint_mode epochs
```
4. To produce model predictions on a directory of test images run script predict.py with the following arguments :
```
python predict.py --image_size 256 --num_classes 2 --image_path ./test --model_path ./model.h5 --output_path ./results
