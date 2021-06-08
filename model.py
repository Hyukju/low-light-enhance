import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np 
import utils
import cv2

crop_width = 256
crop_height = 256

def data_generator_from_path(x_train, y_train, batch_size=4, use_shuffle=True, use_random_crop=True):
   
    train_full_index = list(x_train.keys())
    train_size = len(train_full_index)
    total_batch = train_size // batch_size - 1

    current_batch = 0
   
    while True:

        if (current_batch > total_batch):
            current_batch = 0
        
        if use_shuffle and current_batch == 0:
                # shuffle train ids
                np.random.shuffle(train_full_index)
                print('batch full_ data shuffling')
        
        current_index = current_batch * batch_size
        batch_mask = train_full_index[current_index:current_index + batch_size]

      
        if use_random_crop:
            x_batch = np.zeros((batch_size, crop_height, crop_width ,3))
            y_batch = np.zeros((batch_size, crop_height, crop_width ,3))

            for i in range(batch_size):  
                batch_mask_index = batch_mask[i]              
                x_img = cv2.imread(x_train[batch_mask_index]) / 255.0
                y_img = cv2.imread(y_train[batch_mask_index]) / 255.0
                crop_x, crop_y = utils.get_random_crop_coordinate(x_img, crop_width, crop_height)
                cropped_x_img = utils.crop_image(x_img, crop_x, crop_y, crop_width, crop_height)
                cropped_y_img = utils.crop_image(y_img, crop_x, crop_y, crop_width, crop_height)

                # ----------------
                # augmentation?
                # ----------------
                x_batch[i] = cropped_x_img
                y_batch[i] = cropped_y_img
        else:
            x_batch = cv2.imread(x_train[batch_mask])
            y_batch = cv2.imread(y_train[batch_mask])


        yield x_batch, y_batch
        
        current_batch += 1

