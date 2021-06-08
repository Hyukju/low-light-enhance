import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np 
import utils
import cv2

crop_width = 256
crop_height = 256

def data_generator_from_path(x_train, y_train, batch_size=4, use_shuffle=True, use_random_crop=True):
   
    # train_full_index = list(x_train.keys())
    train_full_index = len(x_train)
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


def buil_model():

    input_img = keras.Input(shape=(None, None, 3))

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)

    return autoencoder

def train():

    x_data, y_data = utils.load_dataset_sice()
    
    data_keys = list(x_data.keys())

    length = len(x_data)
    num_train = int(length * 0.7)
    num_val = int(length * 0.9)

    x_train = [x_data[i] for i in data_keys[:num_train]]
    y_train = [y_data[i] for i in data_keys[:num_train]]

    x_val = [x_data[i] for i in data_keys[num_train: num_val]]
    y_val = [x_data[i] for i in data_keys[num_train: num_val]]

    x_test = [x_data[i] for i in data_keys[num_val: -1]]
    y_test = [y_data[i] for i in data_keys[num_val: -1]]

    train_generator = data_generator_from_path(x_train, y_train)
    val_generator = data_generator_from_path(x_val, y_val)

    model = buil_model()
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    model.fit_generator(train_generator, steps_per_epoch=10, epochs=3, validation_data=val_generator,validation_steps=3)





