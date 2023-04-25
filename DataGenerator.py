"""
Author: Jamie McGrath

This file contains a class `DataGenerator` for creating data generators using 
`ImageDataGenerator` from Keras for training, validation, and test data for 
image classification. The `DataGenerator` class provides methods `train_gen()`, 
`valid_gen()`, and `test_gen()` to create data generators for respective datasets, 
with appropriate preprocessing and augmentation options.

Params:
~ `train_gen()`:
    data generator for training data with preprocessing and augmentation options.
~ `valid_gen()`:
    data generator for validation data with preprocessing options.
~ `test_gen()`:
    data generator for test data with preprocessing options.

Note: The image data must be organised in the following directory structure:
~ 'archive/images/1.train': Directory containing training images
~ 'archive/images/2.validation': Directory containing validation images
~ 'archive/images/3.test': Directory containing test images
"""

from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

class DataGenerator:
    
    def __init__(self):
        """
        Constructor for the DataGenerator class.

        Default Values:
            batch_size (int): batch size (64) used for generating data batches.
            train_dir (str): path to the data directory "archive/images/1.train".
            valid_dir (str): path to the data directory "archive/images/2.validation".
            test_dir (str): path to the data directory "archive/images/3.test".
        """
        self.batch_size = 64
        self.train_dir = Path("archive") / "images" / "1.train"
        self.valid_dir = Path("archive") / "images" / "2.validation"
        self.test_dir = Path("archive") / "images" / "3.test"
        
    def train_gen(self):
        """
        Create a data generator for training data.

        Returns:
            train_data (ImageDataGenerator.flow_from_directory): 
                Data generator for training data.
        """
        train_data = ImageDataGenerator(
            rescale=1.0 / 255.0, # rescale pixel values to range [0,1]
            width_shift_range=0.1, # shift images horizontally by up to 10% of image width
            height_shift_range=0.1, # shift images vertically by up to 10% of image height
            rotation_range=20, # rotate images by up to 20 degrees
            shear_range=0.1, # apply shearing transformation with intensity of 0.1
            zoom_range=0.1, # zoom in on images by up to 10%
            fill_mode='nearest' # fill missing pixels with nearest pixel values
        )
        return train_data.flow_from_directory(
            self.train_dir, # directory containing training images
            target_size=(48, 48), # resize images to size 48x48
            color_mode="grayscale", # convert images to grayscale
            batch_size=self.batch_size, # number of images in each batch
            class_mode="categorical", # type of label generation for classification
            seed=42, # seed for random number generator for reproducibility
            shuffle=True, # shuffle the images after each epoch
        )
        
    def valid_gen(self):
        """
        Create a data generator for validation data.

        Returns:
            valid_data (ImageDataGenerator.flow_from_directory): 
                Data generator for validation data.
        """
        valid_data = ImageDataGenerator(rescale=1.0 / 255) # rescale pixel values to range [0,1]
        return valid_data.flow_from_directory(
            self.valid_dir, # directory containing validation images
            target_size=(48, 48), # resize images to size 48x48
            color_mode="grayscale", # convert images to grayscale
            batch_size=self.batch_size, # number of images in each batch
            class_mode="categorical", # type of label generation for classification
            shuffle=False, # do not shuffle the images
        )
        
    def test_gen(self):
        """
        Create a data generator for test data.

        Returns:
            test_data (ImageDataGenerator.flow_from_directory): 
                Data generator for test data.
        """
        test_data = ImageDataGenerator(rescale=1.0 / 255) # rescale pixel values to range [0,1]
        return test_data.flow_from_directory(
            self.test_dir, # directory containing test images
            target_size=(48, 48), # resize images to size 48x48
            color_mode="grayscale", # convert images to grayscale
            batch_size=self.batch_size, # number of images in each batch
            class_mode="categorical", # type of label generation for classification
            shuffle=False, # do not shuffle the images
        )
