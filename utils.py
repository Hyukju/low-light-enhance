import numpy as np 
import os 

def resize_image(img, resize_width, resize_height, mode='square'):
    assert mode in ['square', 'crop']
    pass

def get_random_crop_coordinate(img, crop_width, crop_height):
    assert img.shape[0] >= crop_height
    assert img.shape[1] >= crop_width
    x = np.random.randint(0, img.shape[1] - crop_width)
    y = np.random.randint(0, img.shape[0] - crop_height)
    return x, y

def crop_image(img, crop_x, crop_y, crop_width, crop_height):
    return img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width, :]

def load_dataset_sice():

    dataset_dir = 'D:\\datasets\\SICE\\Dataset_Part1\\'
    dir_list = os.listdir(dataset_dir)
    dir_list.remove('Label')

    x_train_dir_list = dir_list
    y_train_dir_list = ['Label']

    x_train = {}
    y_train = {}

    for dir_name in x_train_dir_list:    
        img_dir_path = os.path.join(dataset_dir, dir_name)
        for file_name in os.listdir(img_dir_path):
            file_path = os.path.join(img_dir_path, file_name)
            if os.path.isfile(file_path):
                x_img_path = file_path
                y_img_path = os.path.join(dataset_dir, y_train_dir_list[0], dir_name + '.jpg')
                x_train[dir_name] = x_img_path
                y_train[dir_name] = y_img_path

    return x_train, y_train