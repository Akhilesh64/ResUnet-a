# ResUnet-a

1. Change paths in savetif2npy.py and run teh script to change the image data to numpy arrays.
2. Run preprocess_save_patches_ISPRS.py for patching the tiles, normalizing the images and 1-hot encode the label data. Provide arg --num_classes for the no. of classes(2 in our case).
3. Change the paths and hyperparameters in main.py and run it to start model training.
4. Finally run predict.py and provide test set path to produce model predictions.
