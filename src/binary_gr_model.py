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

imgs = pathlib.Path(r'C:\Users\nghug\Desktop\AI_Research\Images_Resized_3')
batch_size = 32
img_height = 300
img_width = 400
AUTOTUNE = tf.data.AUTOTUNE

def get_data() -> Union[Any, Any, Any]:
    """
    Fetches the data from the global Path variable imgs
    The seed is randomized, and we use a validation split of 0.1
    :return: The training and validation datasets, and the class names
    """
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
    """
    Train the model and save it
    :param verbose: Boolean to output model summary
    :return: None
    """

    # Get the data, shuffle it, and prepare to fetch
    train_ds, val_ds, class_names = get_data()
    train_ds = train_ds.cache().shuffle(3200).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data augmentation helps prevent over-fitting and the model memorizing the training data
    # We use random horizontal flips, rotations, zooms, and contrast, since these would be the
    # most realistic differences in real data
    data_augmentation = keras.Sequential([
        RandomFlip('horizontal', input_shape=(img_height, img_width, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomContrast(0.1)
    ])

    # The Sequential model we are using
    # We first add our data augmentation model, so that way each epoch the data gets augmented.
    # We then normalize the pixel data with Rescaling, and then pass the training data into
    # learning layers of Conv2D and MaxPooling2D. Since we had issues with over-fitting,
    # we added a Dropout layer of 0.1 which randomly sets inputs to 0, while scaling up
    # the other inputs by 1/(1-0.1) = 1.11. We then Flatten the data and pass it through
    # a Dense layer, and then a final Dense layer, which sorts the data into one of the
    # six provided classes: 3L, 4L, 5L+, Bilayer, Monolayer, or NoSample.
    model = Sequential([
        data_augmentation,
        Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.1),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(6)  # number of classes = 6
    ])

    # Compile the model. We use the default Adam optimizer because it works well and its default learning
    # rate of 0.001 provides an ideal curve for the accuracy and loss of the training data. Since we are
    # working with multiple categories, our loss must be calculated by SparseCategoricalCrossentropy
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose:
        model.summary()

    # Fitting the model over 50 epochs (iterations)
    epochs = 50
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    # Fetching accuracy and loss history for both training and validation
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
    plt.savefig('saved_figs/binary_graphene_model_4_figs')
    plt.show()

    # Save the model
    model.save('saved_models/binary_graphene_model_4')

def predict() -> None:
    model = keras.models.load_model(r'C:\Users\nghug\Desktop\AI_Research\src\saved_models\graphene_model_4')
    img = keras.utils.load_img(
        r'C:\Users\nghug\Desktop\AI_Research\Max\MAX5-1a_image2_60x-2-20pm_5-16-23.jpg', target_size=(img_height, img_width)
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


if __name__ == '__main__':
    main(verbose=True)
