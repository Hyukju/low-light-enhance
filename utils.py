import numpy as np 
import os 
import cv2
import time 

def get_local_time():    
    return time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

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

def load_dataset_sice(dataset_dir, dataset_type):
    assert dataset_type in ['path', 'image']

    # dataset_dir= 'D:\\datasets\\SICE\\Dataset_Part1\\'
    dir_list = os.listdir(dataset_dir)
    dir_list.remove('Label')

    x_data_dir_list = dir_list
    y_data_dir_list = ['Label']

    x_data = {}
    y_data = {}

    for dir_name in x_data_dir_list:    
        img_dir_path = os.path.join(dataset_dir, dir_name)
        for file_name in os.listdir(img_dir_path):
            file_path = os.path.join(img_dir_path, file_name)
            if os.path.isfile(file_path):
                x_img_path = file_path
                y_img_path = os.path.join(dataset_dir, y_data_dir_list[0], dir_name + '.jpg')
               
                if dataset_type == 'path':                    
                    x_data[dir_name] = x_img_path
                    y_data[dir_name] = y_img_path
                elif dataset_type == 'image':
                    x_data[dir_name] = cv2.imread(x_img_path)
                    y_data[dir_name] = cv2.imread(y_img_path)

    return x_data, y_data

