import keras.utils
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import keras.layers.merging.concatenate as Concatenate

img_height = 300
img_width = 400
K = 6

def wnet_model(verbose: bool = False, plot: bool = False) -> None:
    inputs = Input((img_height, img_width, 3))

    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)

    x2 = MaxPooling2D((2, 2), padding='same')(x1)
    x2 = SeparableConv2D(128, (2, 2), activation='relu', padding='same')(x2)
    x2 = SeparableConv2D(128, (2, 2), activation='relu', padding='same')(x2)

    x3 = MaxPooling2D((2, 2), padding='same')(x2)
    x3 = SeparableConv2D(256, (2, 2), activation='relu', padding='same')(x3)
    x3 = SeparableConv2D(256, (2, 2), activation='relu', padding='same')(x3)

    x4 = MaxPooling2D((2, 2), padding='same')(x3)
    x4 = SeparableConv2D(512, (2, 2), activation='relu', padding='same')(x4)
    x4 = SeparableConv2D(512, (2, 2), activation='relu', padding='same')(x4)

    x5 = MaxPooling2D((2, 2), padding='same')(x4)
    x5 = SeparableConv2D(1024, (2, 2), activation='relu', padding='same')(x5)
    x5 = SeparableConv2D(1024, (2, 2), activation='relu', padding='same')(x5)

    x6 = Conv2DTranspose(1024, kernel_size=(2, 2), strides=2, activation='relu', padding='same')(x5)
    x6 = Concatenate([x6, x4])
    x6 = SeparableConv2D(512, (2, 2), activation='relu', padding='same')(x6)
    x6 = SeparableConv2D(512, (2, 2), activation='relu', padding='same')(x6)

    x7 = Conv2DTranspose(512, (2, 2), strides=2, activation='relu', padding='same')(x6)
    x7 = Cropping2D(cropping=((1, 0), (0, 0)))(x7)
    x7 = concatenate([x7, x3])
    x7 = SeparableConv2D(256, (2, 2), activation='relu', padding='same')(x7)
    x7 = SeparableConv2D(256, (2, 2), activation='relu', padding='same')(x7)

    x8 = Conv2DTranspose(256, (2, 2), strides=2, activation='relu', padding='same')(x7)
    x8 = concatenate([x8, x2])
    x8 = SeparableConv2D(128, (2, 2), activation='relu', padding='same')(x8)
    x8 = SeparableConv2D(128, (2, 2), activation='relu', padding='same')(x8)

    x9 = Conv2DTranspose(128, (2, 2), strides=2, activation='relu', padding='same')(x8)
    x9 = concatenate([x9, x1])
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same')(x9)
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same')(x9)
    x9 = Conv2D(K, (1, 1), activation='relu', padding='same')(x9)
    x9 = Softmax()(x9)

    x10 = Conv2D(64, (3, 3), activation='relu', padding='same')(x9)
    x10 = Conv2D(64, (3, 3), activation='relu', padding='same')(x10)

    x11 = MaxPooling2D((2, 2), padding='same')(x10)
    x11 = SeparableConv2D(128, (2, 2), activation='relu', padding='same')(x11)
    x11 = SeparableConv2D(128, (2, 2), activation='relu', padding='same')(x11)

    x12 = MaxPooling2D((2, 2), padding='same')(x11)
    x12 = SeparableConv2D(256, (2, 2), activation='relu', padding='same')(x12)
    x12 = SeparableConv2D(256, (2, 2), activation='relu', padding='same')(x12)

    x13 = MaxPooling2D((2, 2), padding='same')(x12)
    x13 = SeparableConv2D(512, (2, 2), activation='relu', padding='same')(x13)
    x13 = SeparableConv2D(512, (2, 2), activation='relu', padding='same')(x13)

    x14 = MaxPooling2D((2, 2), padding='same')(x13)
    x14 = SeparableConv2D(1024, (2, 2), activation='relu', padding='same')(x14)
    x14 = SeparableConv2D(1024, (2, 2), activation='relu', padding='same')(x14)

    x15 = Conv2DTranspose(1024, (2, 2), strides=2, activation='relu', padding='same')(x14)
    x15 = concatenate([x15, x13])
    x15 = SeparableConv2D(512, (2, 2), activation='relu', padding='same')(x15)
    x15 = SeparableConv2D(512, (2, 2), activation='relu', padding='same')(x15)

    x16 = Conv2DTranspose(512, (2, 2), strides=2, activation='relu', padding='same')(x15)
    x16 = Cropping2D(cropping=((1, 0), (0, 0)))(x16)
    x16 = concatenate([x16, x12])
    x16 = SeparableConv2D(256, (2, 2), activation='relu', padding='same')(x16)
    x16 = SeparableConv2D(256, (2, 2), activation='relu', padding='same')(x16)

    x17 = Conv2DTranspose(256, (2, 2), strides=2, activation='relu', padding='same')(x16)
    x17 = concatenate([x17, x11])
    x17 = SeparableConv2D(128, (2, 2), activation='relu', padding='same')(x17)
    x17 = SeparableConv2D(128, (2, 2), activation='relu', padding='same')(x17)

    x18 = Conv2DTranspose(128, (2, 2), strides=2, activation='relu', padding='same')(x17)
    x18 = concatenate([x18, x10])
    x18 = Conv2D(64, (3, 3), activation='relu', padding='same')(x18)
    x18 = Conv2D(64, (3, 3), activation='relu', padding='same')(x18)
    x18 = Conv2D(3, (1, 1), activation='relu', padding='same')(x18)

    wnet = Model(inputs, x18)
    wnet.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if verbose:
        wnet.summary()
    if plot:
        keras.utils.plot_model(wnet, to_file='/Users/firsttry/Desktop/wnet.png', show_shapes=True)


wnet_model(True, True)