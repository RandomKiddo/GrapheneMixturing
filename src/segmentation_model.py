import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output
from typing import *

from keras import Model
from keras.applications import MobileNetV2

img_width = 400
img_height = 300

def main() -> None:
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
        pix2pix.upsample(512, 3)
    ]


if __name__ == '__main__':
    main()
