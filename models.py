import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np 
import utils
import cv2

crop_width = 256
crop_height = 256

# def data_generator_from_path(x_train, y_train, batch_size=4, use_shuffle=True, use_random_crop=True):
   
#     # train_full_index = list(x_train.keys())
#     train_size = len(x_train)
#     train_full_index = np.arange(train_size)
#     total_batch = train_size // batch_size - 1

#     current_batch = 0
   
#     while True:

#         if (current_batch > total_batch):
#             current_batch = 0
        
#         if use_shuffle and current_batch == 0:
#                 # shuffle train ids
#                 np.random.shuffle(train_full_index)
#                 #print('batch full_ data shuffling')
        
#         current_index = current_batch * batch_size
#         batch_mask = train_full_index[current_index:current_index + batch_size]

      
#         if use_random_crop:
#             x_batch = np.zeros((batch_size, crop_height, crop_width ,3))
#             y_batch = np.zeros((batch_size, crop_height, crop_width ,3))

#             for i in range(batch_size):  
#                 batch_mask_index = batch_mask[i]              
#                 x_img = cv2.imread(x_train[batch_mask_index]) / 255.0
#                 y_img = cv2.imread(y_train[batch_mask_index]) / 255.0
#                 crop_x, crop_y = utils.get_random_crop_coordinate(x_img, crop_width, crop_height)
#                 cropped_x_img = utils.crop_image(x_img, crop_x, crop_y, crop_width, crop_height)
#                 cropped_y_img = utils.crop_image(y_img, crop_x, crop_y, crop_width, crop_height)

#                 # ----------------
#                 # augmentation?
#                 # ----------------
#                 x_batch[i] = cropped_x_img
#                 y_batch[i] = cropped_y_img
#         else:
#             x_batch = cv2.imread(x_train[batch_mask])
#             y_batch = cv2.imread(y_train[batch_mask])


#         yield x_batch, y_batch
        
#         current_batch += 1


def data_generator_from_path(x_train, y_train, batch_size=4, use_random_crop=True):
   
    keys = list(x_train.keys())
    resize_width, resize_height = crop_width, crop_height
   
    while True:
        # data shuffling               
        batch_keys = np.random.choice(keys, batch_size, replace=False)       

        x_batch = np.zeros((batch_size, crop_height, crop_width ,3))
        y_batch = np.zeros((batch_size, crop_height, crop_width ,3))

        for i, key in enumerate(batch_keys):                              
            x_img = cv2.imread(x_train[key]) 
            y_img = cv2.imread(y_train[key]) 
            if use_random_crop:         
                crop_x, crop_y = utils.get_random_crop_coordinate(x_img, crop_width, crop_height)
                xx_img = utils.crop_image(x_img, crop_x, crop_y, crop_width, crop_height)
                yy_img = utils.crop_image(y_img, crop_x, crop_y, crop_width, crop_height)
            else:
                # image resize
                xx_img =cv2.resize(x_img, dsize=(resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
                yy_img =cv2.resize(y_img, dsize=(resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
                

            # ----------------
            # augmentation?
            # ----------------
            x_batch[i] = xx_img / 255.0
            y_batch[i] = yy_img / 255.0

        yield x_batch, y_batch


def buil_model(height, width, channel):

    input_img = keras.Input(shape=(height, width, channel))

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img) 
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) 
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x) 
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)   
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)

    autoencoder.summary()

    return autoencoder

def split_dictionary(dict_data, split_key_list):
    splited_dict = {}
    for key in split_key_list:
        splited_dict[key] = dict_data[key]
    return splited_dict


def train(epochs=100, batch_size=10):    

    x_data, y_data = utils.load_dataset_sice()
    
    data_keys = list(x_data.keys())

    num_data = len(x_data)
    last_index_train = int(num_data * 0.7)
    last_index_val = int(num_data * 0.9)

    x_train = split_dictionary(x_data, data_keys[:last_index_train])
    y_train = split_dictionary(y_data, data_keys[:last_index_train])

    x_val = split_dictionary(x_data,  data_keys[last_index_train: last_index_val])
    y_val = split_dictionary(y_data,  data_keys[last_index_train: last_index_val])

    x_test = split_dictionary(x_data,  data_keys[last_index_val: -1])
    y_test = split_dictionary(x_data,  data_keys[last_index_val: -1])

    train_generator = data_generator_from_path(x_train, y_train, batch_size=batch_size)
    val_generator = data_generator_from_path(x_val, y_val, batch_size=batch_size)

    #
    num_train_data = len(x_train)
    num_val_data = len(x_val)

    print('-------------------------------------')
    print('num of train data = ', num_train_data)
    print('num of val data = ', num_train_data)
    print('batch size = ', batch_size)
    print('epochs = ', epochs)
    print('-------------------------------------')

    model = buil_model(None, None, 3)
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    model.fit_generator(train_generator, 
                        steps_per_epoch=num_train_data//batch_size, 
                        epochs=epochs, 
                        validation_data=val_generator,
                        validation_steps=num_val_data//batch_size,
                        callbacks=[keras.callbacks.ModelCheckpoint(filepath='./weights/my_model.{epoch:02d}-{val_loss:.2f}.hdf5',verbose=0, save_weights_only=True, period=20)])
    
    return model, x_test, y_test



if __name__=='__main__':
    train()
    
    





