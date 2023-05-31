import keras
from IPython.core.display_functions import clear_output
from keras import Model
from keras.applications import MobileNetV2
from keras.callbacks import Callback
from keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.data import Dataset
from tensorflow_examples.models.pix2pix import pix2pix
from keras.layers import *
import numpy as np
from PIL import Image
import os
from typing import *
import tensorflow as tf

img_width = 400
img_height = 300
batch_size = 32

def npz_data() -> None:
    path_masks = r'C:\Users\nghug\Desktop\AI_Research\Segmentation\Masks'
    path_normal = r'C:\Users\nghug\Desktop\AI_Research\Segmentation\Normal'

    array_masks = []
    array_normal = []

    for f in os.listdir(path_normal):
        single_img = Image.open(path_normal + '\\' + f)
        single_array = np.array(single_img)
        array_normal.append(single_array)
    for f in os.listdir(path_masks):
        single_img = Image.open(path_masks + '\\' + f)
        single_array = np.array(single_img)
        array_masks.append(single_array)

    last = int(len(array_masks))

    np.savez(r'C:\Users\nghug\Desktop\AI_Research\segmentation_dataset.npz', x_train=array_normal[:last],
             y_train=array_masks[:last], x_valid=array_normal[last:], y_valid=array_masks[last:])

def get_data() -> tuple[Dataset, Dataset]:
    with np.load(r'C:\Users\nghug\Desktop\AI_Research\segmentation_dataset.npz') as data:
        train_x = data['x_train']
        train_y = data['y_train']
        valid_x = data['x_valid']
        valid_y = data['y_valid']
    return Dataset.from_tensor_slices((train_x, train_y)), Dataset.from_tensor_slices((valid_x, valid_y))

def model(verbose: bool = False) -> None:
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False)
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False
    up_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64, 3)
    ]
    inputs = Input(shape=(img_height, img_width, 3))
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])
    last = Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same')  # 2 classes / output channels
    x = last(x)
    u_net = Model(inputs=inputs, outputs=x)
    u_net.compile(optimizer='Adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    if verbose:
        u_net.summary()
        keras.utils.plot_model(u_net, show_shapes=True)

    train_ds, val_ds = get_data()  # todo work with train_ds
    train_ds = train_ds.map()

    # todo continue
    epochs = 20
    val_subsplits = 5

    history = u_net.fit()

if __name__ == '__main__':
    pass
