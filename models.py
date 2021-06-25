import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np 
import utils
import cv2
import os 

crop_width = 256
crop_height = 256

def data_generator(x_train, y_train, dataset_type='path', batch_size=4, use_random_crop=True):
    assert dataset_type in ['path', 'image']

    keys = list(x_train.keys())
    resize_width, resize_height = crop_width, crop_height
   
    while True:
        # data shuffling               
        batch_keys = np.random.choice(keys, batch_size, replace=False)       

        x_batch = np.zeros((batch_size, crop_height, crop_width ,3))
        y_batch = np.zeros((batch_size, crop_height, crop_width ,3))

        for i, key in enumerate(batch_keys):                              
            if dataset_type == 'path':
                x_img = cv2.imread(x_train[key]) 
                y_img = cv2.imread(y_train[key]) 
            elif dataset_type =='image':
                x_img = x_train[key].copy()
                y_img = y_train[key].copy()

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
            x_batch[i] = xx_img.astype('float32') / 255.0
            y_batch[i] = yy_img.astype('float32') / 255.0

        yield x_batch, y_batch

def build_model(height, width, channel):

    input_layer = keras.Input(shape=(height, width, channel))

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer) 
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) 

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)   
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2))(x)    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D((2, 2))(x)   
    x = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    output_layer = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


    autoencoder = keras.Model(input_layer, output_layer)
    autoencoder.summary()

    return autoencoder

def build_skip_model(height, width, channel):

    input_layer = keras.Input(shape=(height, width, channel))
    # down
    x0 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer) 
    x0 = layers.BatchNormalization()(x0)
    x0 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x0)
    x0 = layers.BatchNormalization()(x0)
    p0 = layers.MaxPooling2D((2, 2), padding='same')(x0) 

    x1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p0) 
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.BatchNormalization()(x1)
    p1 = layers.MaxPooling2D((2, 2), padding='same')(x1)

    x2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    p2 = layers.MaxPooling2D((2, 2), padding='same')(x2)

    x3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x3)   
    x3 = layers.BatchNormalization()(x3)
    p3 = layers.MaxPooling2D((2, 2), padding='same')(x3)
    # latent
    x4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    # up
    u0 = layers.UpSampling2D((2, 2))(x4)
    x5 = layers.concatenate([u0, x3])
    x5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x5)
    x5 = layers.BatchNormalization()(x5)

    u1 = layers.UpSampling2D((2, 2))(x5)
    x6 = layers.concatenate([u1, x2])
    x6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x6)
    x6 = layers.BatchNormalization()(x6)
    x6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x6)
    x6 = layers.BatchNormalization()(x6)

    u2 = layers.UpSampling2D((2, 2))(x6)    
    x7 = layers.concatenate([u2, x1])
    x7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x7)
    x7 = layers.BatchNormalization()(x7)
    x7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x7)
    x7 = layers.BatchNormalization()(x7)

    u3 = layers.UpSampling2D((2, 2))(x7)   
    x8 = layers.concatenate([u3, x0])
    x8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x8)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x8)
    x8 = layers.BatchNormalization()(x8)
    
    x8 = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x8)
    output_layer = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x8)

    autoencoder = keras.Model(input_layer, output_layer)
    autoencoder.summary()

    return autoencoder

def split_dictionary(dict_data, split_key_list):
    splited_dict = {}
    for key in split_key_list:
        splited_dict[key] = dict_data[key]
    return splited_dict


def train(dataset_dir, dataset_type='path', epochs=100, batch_size=10, use_random_crop=True, loss='mae', save_weight_dir=None):    

    assert dataset_type in ['path', 'image']
    
    x_data, y_data = utils.load_dataset_sice(dataset_dir, dataset_type)
    
    data_keys = list(x_data.keys())

    num_data = len(x_data)
    last_index_train = int(num_data * 0.7)
    last_index_val = int(num_data * 0.9)

    x_train = split_dictionary(x_data, data_keys[:last_index_train])
    y_train = split_dictionary(y_data, data_keys[:last_index_train])

    x_val = split_dictionary(x_data,  data_keys[last_index_train: last_index_val])
    y_val = split_dictionary(y_data,  data_keys[last_index_train: last_index_val])

    x_test = split_dictionary(x_data,  data_keys[last_index_val: -1])
    y_test = split_dictionary(y_data,  data_keys[last_index_val: -1])

    train_generator = data_generator(x_train, y_train, dataset_type, batch_size=batch_size, use_random_crop=use_random_crop)
    val_generator = data_generator(x_val, y_val, dataset_type, batch_size=batch_size, use_random_crop=use_random_crop)
    

    #
    num_train_data = len(x_train)
    num_val_data = len(x_val)

    print('-------------------------------------')
    print('num of train data = ', num_train_data)
    print('num of val data = ', num_val_data)
    print('batch size = ', batch_size)
    print('epochs = ', epochs)
    print('-------------------------------------')

    try:
        # make weights dir
        if save_weight_dir == None:
            local_time = utils.get_local_time()
            os.makedirs(os.path.join('./weights',local_time))
        else:
            os.makedirs(os.path.join('./weights',save_weight_dir))
    except:
        pass

    save_weight_file = os.path.join('./weights', save_weight_dir, 'my_model.{epoch:02d}.h5')


    model = build_skip_model(None, None, 3)
    model.compile(optimizer='adam', loss=loss, metrics=['acc'])
    model.fit_generator(train_generator, 
                        steps_per_epoch=num_train_data//batch_size, 
                        epochs=epochs, 
                        validation_data=val_generator,
                        validation_steps=num_val_data//batch_size,
                        callbacks=[keras.callbacks.ModelCheckpoint(filepath=save_weight_file,verbose=0, save_weights_only=True, period=100)])
    
    return model, x_test, y_test


if __name__=='__main__':
    dataset_dir = 'D:\\projects\\_datasets\\SICE\\Dataset_Part1\\'    
    train(dataset_dir, dataset_type='image', epochs=1000, batch_size=16,use_random_crop=False, loss= 'mae',save_weight_dir='rgb_mae_model2_resize')    
    train(dataset_dir, dataset_type='image', epochs=1000, batch_size=16,use_random_crop=False, loss= 'mse',save_weight_dir='rgb_mse_model2_resize')    
    
    





