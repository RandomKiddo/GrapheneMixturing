import keras
import tensorflow as tf
from keras import *
from keras.layers import *
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory
import pathlib
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

imgs = pathlib.Path(r'C:\Users\nghug\PycharmProjects\AIConda\master\Images_Resized')
batch_size = 32  # may need to reduce based on CPU and GPU specs
img_height = 300
img_width = 400
AUTOTUNE = tf.data.AUTOTUNE

def get_data() -> Union[Any, Any, Any]:
    '''
    Fetches the data from the global Path variable imgs
    :return: The training and validation datasets, and the class names
    '''
    train_ds = image_dataset_from_directory(
        imgs,
        validation_split=0.1,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = image_dataset_from_directory(
        imgs,
        validation_split=0.1,
        subset='validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    return train_ds, val_ds, train_ds.class_names

def main(verbose: bool = False) -> None:
    '''
    Train the model and save it
    :param verbose: Boolean to output model summary
    :return: None
    '''
    train_ds, val_ds, class_names = get_data()
    train_ds = train_ds.cache().shuffle(3200).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = keras.Sequential([
        RandomFlip('horizontal', input_shape=(img_height, img_width, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1)
    ])

    model = Sequential([
        data_augmentation,
        Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.1),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(6)  # number of classes = 6
    ])
    # try more epochs, greater learning rate, or sgd optimizer
    model.compile(optimizer='Adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    if verbose:
        model.summary()

    epochs = 25
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    model.save('saved_models/graphene_model_3')

def test() -> None:
    model = keras.models.load_model('/Users/firsttry/Desktop/Git/summer/master/src/saved_models/graphene_model')
    img = keras.utils.load_img(
        '/Users/firsttry/Desktop/Git/summer/master/Images/Monolayer/Gr-15-8-mono-100x.jpg', target_size=(img_height, img_width)
    )
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    train_ds, val_ds, class_names = get_data()
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

def augment_and_save_images() -> None:
    '''
    This function should only be called once. It will multiply the dataset ten-fold through augmentation
    :return: None
    '''
    data_augmentation = keras.Sequential([
        RandomFlip('horizontal', input_shape=(img_height, img_width, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomContrast(0.1)
    ])
    no_repeat_number = 0
    ds = image_dataset_from_directory(
        imgs,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=1
    )

def get_label_from_int(i: int) -> str:
    labels = ('3L', '4L', '5L+', 'Bilayer', 'Monolayer', 'NoSample')
    return labels[i]

def resize_data() -> None:
    subdirs = ('3L', '4L', '5L+', 'Bilayer', 'Monolayer', 'NoSample')
    PATH = r'C:\Users\nghug\PycharmProjects\AIConda\master\Images'
    for i in range(6):
        for f in os.listdir(PATH + '\\' + subdirs[i]):
            img = Image.open(PATH + '\\' + subdirs[i] + '\\' + f)
            resized = img.resize((400, 300))
            resized.save(r'C:\Users\nghug\PycharmProjects\AIConda\master\Images_Resized_2' + '\\' + subdirs[i] + '\\' + f)


if __name__ == '__main__':
    main(verbose=True)  # todo verbose=False
    # test()
    # augment_and_save_images()
    # resize_data()

