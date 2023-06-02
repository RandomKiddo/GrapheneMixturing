import keras
from IPython.core.display_functions import clear_output
from keras import Model
from keras.applications import MobileNetV2
from keras.callbacks import Callback, LearningRateScheduler
from keras.losses import SparseCategoricalCrossentropy, categorical_crossentropy
from keras.optimizers import Adam
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

def double_conv_block(x: Any, n_filters: Any) -> Any:
    x = Conv2D(n_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(n_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x

def downsample_block(x: Any, n_filters: Any) -> Tuple[Any, Any]:
    f = double_conv_block(x, n_filters)
    p = MaxPool2D(strides=2, padding='same')(f)
    p = Dropout(0.2)(p)
    return f, p

def upsample_block(x: Any, conv_features: Any, n_filters: Any) -> Any:
    x = Conv2DTranspose(n_filters, 3, 2, padding='same')(x)
    x = Cropping2D(cropping=((x.shape[1]-conv_features.shape[1], 0), (x.shape[2]-conv_features.shape[2], 0)))(x)
    x = concatenate([x, conv_features])
    x = Dropout(0.2)(x)
    x = double_conv_block(x, n_filters)
    return x

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

    inputs = Input(shape=(img_height, img_width, 3))

    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    bottleneck = double_conv_block(p4, 1024)

    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = Conv2D(2, 1, padding='same', activation='softmax')(u9)

    u_net = Model(inputs, outputs, name='U-Net')
    u_net.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics='accuracy')

    if verbose:
        u_net.summary()
    if plot:
        plot_model(u_net, to_file=r'/Users/firsttry/Desktop/u_net.jpg', show_shapes=True)

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

    epochs = 25
    history = u_net.fit(train_batches, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_batches)

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
    plt.savefig('/Users/firsttry/Desktop/u_net_model_6_figs.png')
    plt.show()

    # Save the model
    u_net.save('/Users/firsttry/Desktop/u_net_model_6')

def predict() -> None:
    model = keras.models.load_model(r'/Users/firsttry/Desktop/u_net_model_5')
    img = keras.utils.load_img(
        r'/Users/firsttry/Desktop/Segmentation/Normal/137_3.1.jpg',
        target_size=(img_height, img_width)
    )
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    mask = model.predict(img_array)
    pred_mask = tf.math.argmax(mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    plt.imshow(pred_mask[0])
    plt.show()

def test() -> None:
    def n(input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    def l(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        print(datapoint['segmentation_mask'])
        input_mask = tf.image.resize(
            datapoint['segmentation_mask'],
            (128, 128),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )

        input_image, input_mask = n(input_image, input_mask)

        return input_image, input_mask

    def d(display_list):
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    print(info)
    train_images = dataset['train'].map(l, num_parallel_calls=tf.data.AUTOTUNE)
    train_batches = (train_images.cache().shuffle(1000).batch(64).repeat().prefetch(buffer_size=tf.data.AUTOTUNE))
    for image, mask in train_batches.take(1):
        with open(r'C:\Users\nghug\Desktop\AI_Research\output.txt', 'w') as f:
            print(mask, file=f)
        f.close()


if __name__ == '__main__':
    # solidify_masks()
    # npz_data()
    model(verbose=False, plot=True)
    # predict()
