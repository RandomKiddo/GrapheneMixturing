import tensorflow as tf
import keras
import numpy as np
from gmm_clustering import process
import matplotlib.pyplot as plt

# todo fix 'PATH'

BINARY_CLASS_NAMES = ['NoSample', 'Sample']

def exec_ip() -> None:
    model = keras.models.load_model(r'PATH')

    while True:  # todo while end of sample not reached
        # todo fetch_next_img()
        img = keras.utils.load_img(
            'PATH', target_size=(300, 400)
        )
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_name = BINARY_CLASS_NAMES[np.argmax(score)]
        if 'NoSample' not in class_name:
            # todo queue interested
            fp = 'PATH'
            img, db, pt = process(fp)
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title('Main Image')
            plt.subplot(1, 2, 2)
            plt.imshow(db)
            plt.title('Clustered Image')
            plt.show()
            # todo save location
        else:
            # todo delete_img()
            pass

if __name__ == '__main__':
    exec_ip()
