import numpy as np 

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