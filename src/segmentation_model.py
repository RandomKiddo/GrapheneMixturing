import keras
from IPython.core.display_functions import clear_output
from keras import Model
from keras.applications import MobileNetV2
from keras.callbacks import Callback, LearningRateScheduler
from keras.losses import SparseCategoricalCrossentropy, categorical_crossentropy
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from tensorflow.python.data import Dataset
from tensorflow_examples.models.pix2pix import pix2pix
from keras.layers import *
import numpy as np
from PIL import Image
import os
from typing import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pydot
import pydotplus
import graphviz
import tensorflow_datasets as tfds
import keras.backend as K
import random

img_width = 400
img_height = 300
batch_size = 32
buffer_size = 300
train_length = 221
steps_per_epoch = train_length // batch_size
image_colors = [(255, 255, 0), (129, 0, 127)]
color_reference = tf.cast(tf.constant(image_colors), dtype=tf.uint8)

def npz_data() -> None:
    path_masks = r'/Users/firsttry/Desktop/Segmentation/Masks2'
    path_normal = r'/Users/firsttry/Desktop/Segmentation/Normal'

    array_masks = []
    array_normal = []

    for f in os.listdir(path_normal):
        if '._' in f:
            continue
        single_img = Image.open(path_normal + '/' + f)
        single_array = np.array(single_img)
        array_normal.append(single_array)
    for f in os.listdir(path_masks):
        if '._' in f or '.DS_Store' in f:
            continue
        single_img = Image.open(path_masks + '/' + f)
        array = np.ndarray(shape=(img_height, img_width, 1), dtype=int)
        for r in range(single_img.width):
            for c in range(single_img.height):
                R, G, B = single_img.getpixel((r, c))
                if R > 200 and G > 200:
                    array[c, r, 0] = 1
                else:
                    array[c, r, 0] = 0
        save = array
        array_masks.append(array)

    zipped = list(zip(array_normal, array_masks))
    random.shuffle(zipped)

    array_normal, array_masks = list(zip(*zipped))

    last = int(len(array_normal)*0.9)

    np.savez(r'/Users/firsttry/Desktop/segmentation_dataset.npz', x_train=array_normal[:last],
             y_train=array_masks[:last], x_valid=array_normal[last:], y_valid=array_masks[last:])

def solidify_masks() -> None:
    path_masks = r'/Users/firsttry/Desktop/Segmentation/Masks2'
    for f in os.listdir(path_masks):
        if '._' in f or '.DS_Store' in f:
            continue
        img = Image.open(path_masks + '/' + f)
        for r in range(img.width):
            for c in range(img.height):
                R, G, B = img.getpixel((r, c))
                if R > 200 and G > 200:
                    img.putpixel((r, c), (255, 255, 0))
                else:
                    img.putpixel((r, c), (129, 0, 127))
        img.save(path_masks + '/' + f)

def get_data() -> Tuple[Dataset, Dataset]:
    with np.load(r'/Users/firsttry/Desktop/segmentation_dataset.npz') as data:
        train_x = data['x_train']
        train_y = data['y_train']
        valid_x = data['x_valid']
        valid_y = data['y_valid']
    return Dataset.from_tensor_slices((train_x, train_y)), Dataset.from_tensor_slices((valid_x, valid_y))

def normalize(input_image: Any, input_mask: Any) -> Tuple[Any, Any]:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

def load_image(image: Any, mask: Any) -> Tuple[Any, Any]:
    input_image = tf.image.resize(image, (img_height, img_width))
    input_mask = tf.image.resize(
        mask,
        (img_height, img_width),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def add_sample_weights(image: Any, label: Any) -> Tuple[Any, Any, Any]:
    class_weights = tf.constant([1.0, 9.0])
    class_weights = class_weights / tf.reduce_sum(class_weights)
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    return image, label, sample_weights

def model(verbose: bool = False, plot: bool = False) -> None:
    train_ds, val_ds = get_data()  # todo work with train_ds
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    data_augmentation = keras.Sequential([
        RandomContrast(0.1, input_shape=(img_height, img_width, 3))
    ])

    train_batches = (
        train_ds.cache().shuffle(buffer_size).batch(batch_size).repeat()
        .map(lambda x, y: (data_augmentation(x, training=True), y))
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    val_batches = val_ds.batch(batch_size)

    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False)

    layer_names = [
        'block_1_expand_relu',  # 150x200
        'block_3_expand_relu',  # 75x100
        'block_5_expand_relu',  # 38x50
        'block_7_expand_relu',  # 19x25
        'block_13_project'  # 10x13
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

    x = Dropout(.2)(x)

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = keras.layers.Concatenate()
        if x.shape[1]-skip.shape[1] != 0 or x.shape[2]-skip.shape[2] != 0:
            x = Cropping2D(cropping=((x.shape[1]-skip.shape[1], 0), (x.shape[2]-skip.shape[2], 0)))(x)
        x = concat([x, skip])
        x = Dropout(.2)(x)

    last = Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same',
                           kernel_regularizer='l1_l2')  # 2 color classes = 2 filters
    x = last(x)
    x = Dropout(.2)(x)

    u_net = Model(inputs=inputs, outputs=x)
    u_net.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    if verbose:
        u_net.summary()
    if plot:
        plot_model(u_net, to_file=r'/Users/firsttry/Desktop/u_net_3.jpg', show_shapes=True)

    def display(display_list: list) -> None:
        plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    def create_mask(predicted_mask: Any) -> Any:
        predicted_mask = tf.math.argmax(predicted_mask, axis=-1)
        predicted_mask = predicted_mask[..., tf.newaxis]
        return predicted_mask[0]

    def show_predictions(dataset: Any = None, num: int = 1) -> None:
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = model.predict(image)
                display([image[0], mask[0], create_mask(pred_mask)])
        else:
            for images, masks in train_batches.take(1):
                sample_image, sample_mask = images[0], masks[0]
            display([sample_image, sample_mask,
                     create_mask(u_net.predict(sample_image[tf.newaxis, ...]))])

    class DisplayCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)
            show_predictions()
            print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

    show_predictions()

    epochs = 100
    history = u_net.fit(train_batches.map(add_sample_weights), epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_data=val_batches, callbacks=[DisplayCallback()])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    # Visualizing the accuracy plots
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Visualizing the loss plots
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('/Users/firsttry/Desktop/u_net_model_12_figs.png')
    plt.show()

    # Save the model
    u_net.save('/Users/firsttry/Desktop/u_net_model_12')

def predict() -> None:
    model = keras.models.load_model(r'/Users/firsttry/Desktop/u_net_model_10')
    img = keras.utils.load_img(
        r'/Users/firsttry/Desktop/Segmentation/Normal/137_3.1.jpg',
        target_size=(img_height, img_width)
    )
    plt.imshow(img)
    plt.show()
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    mask = model.predict(img_array)
    pred_mask = tf.math.argmax(mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    plt.imshow(pred_mask[0])
    plt.show()


if __name__ == '__main__':
    # solidify_masks()
    # npz_data()
    model(verbose=False, plot=True)
    # predict()
