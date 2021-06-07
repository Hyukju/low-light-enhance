import keras
import matplotlib.pyplot as plt
import numpy as np 
import utils

crop_width = 256
crop_height = 256

def data_generator(x_train, y_train, batch_size=4, use_shuffle=True, use_random_crop=True):
    # x_train and x_train are 4d tensor (none, height, width, channel)
    train_size = x_train.shape[0]
    train_full_index = np.arange(0, train_size)
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
                x_img = x_train[batch_mask_index]
                y_img = y_train[batch_mask_index]
                crop_x, crop_y = utils.get_random_crop_coordinate(x_img, crop_width, crop_height)
                cropped_x_img = utils.crop_image(x_img, crop_x, crop_y, crop_width, crop_height)
                cropped_y_img = utils.crop_image(y_img, crop_x, crop_y, crop_width, crop_height)

                # ----------------
                # augmentation?
                # ----------------
                x_batch[i] = cropped_x_img
                y_batch[i] = cropped_y_img
        else:
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]


        yield x_batch, y_batch
        
        current_batch += 1
