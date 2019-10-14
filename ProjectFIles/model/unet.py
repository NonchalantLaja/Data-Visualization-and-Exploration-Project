import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
from skimage.io import imread, imshow, concatenate_images
from tqdm import tqdm_notebook, tnrange
from keras.models import Model, load_model
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K

K.set_image_data_format('channels_first')
im_width = 32
im_height = 32

# Convolutional Blocks
def contracting(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def expanding(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((3, 3))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def stable(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def Unet():
    f = [4, 8, 16, 32, 64]
    inputs = keras.layers.Input((im_width, im_height, 3), name = 'image')
    p0 = inputs
    c1, p1 = contracting(p0, f[0])
    c2, p2 = contracting(p1, f[1]) 
    c3, p3 = contracting(p2, f[2]) 
    c4, p4 = contracting(p3, f[3]) 
    
    bn = stable(p4, f[4])
    
    u1 = expanding(bn, c4, f[3]) 
    u2 = expanding(u1, c3, f[2]) 
    u3 = expanding(u2, c2, f[1])
    u4 = expanding(u3, c1, f[0]) 
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='lecun_normal')(u4)
    model = keras.models.Model(inputs, outputs)
    return model