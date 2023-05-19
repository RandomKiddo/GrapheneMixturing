from typing import *
import os
import cv2
from keras import *
from keras.layers import *
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIMENSION = 400*300*3
OUTPUT_DIMENSION = 400*300

def generator(filters0: Any, size0: Any) -> Model:
    def add_block(y: Any, filters: Any, size: Any) -> Any:
        y = Conv2DTranspose(filters, size, padding='same', strides=2)(y)
        y = BatchNormalization()(y)
        y = LeakyReLU(0.3)(y)
        return y
    inp = Input(shape=(100,))
    x = Dense(75*100*(filters0*8), input_dim=100)(inp)
    x = BatchNormalization()(x)
    x = Reshape(target_shape=(75, 100, filters0*8))(x)
    x = add_block(x, filters0*4, size0)
    x = add_block(x, filters0*2, size0)
    x = add_block(x, filters0, size0)
    x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
    return Model(inputs=inp, outputs=x, name='Generator')

def discriminator(filters0: Any, size0: Any) -> Model:
    def add_block(y: Any, filters: Any, size: Any) -> Any:
        y = Conv2D(filters, size, padding='same')(y)
        y = BatchNormalization()(y)
        y = Conv2D(filters, size, padding='same', strides=2)(y)
        y = LeakyReLU(0.3)(y)
        return y
    inp = Input(shape=(300, 400, 3))
    x = add_block(inp, filters0, size0)
    x = add_block(x, filters0*2, size0)
    x = add_block(x, filters0*4, size0)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inp, outputs=x, name='Discriminator')

def model(verbose: bool = False) -> Tuple[Model, Model, Sequential]:
    disc = discriminator(16, 5)
    disc.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=.002), metrics=['mae'])
    gen = generator(16, 5)
    gan = Sequential(name='Gan')
    gan.add(gen)
    gan.add(disc)
    disc.trainable = False
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=.002), metrics=['mae'])
    if verbose:
        gen.summary()
        disc.summary()
        gan.summary()
    return gen, disc, gan

def read() -> np.ndarray:
    X = np.empty(shape=(33, 300, 400, 3))
    i = 0
    for f in os.listdir('GANImages'):
        img = cv2.imread('GANImages/' + f)
        if img is None:
            continue
        else:
            X[i] = img
            i += 1
    return X

def train(gen: Sequential, disc: Sequential, gan: Model, epochs: int) -> None:
    if not os.path.isdir('GAN'):
        os.mkdir('GAN')
    avg_loss_disc = []
    avg_loss_gen = []
    total_it = 0
    for epoch in range(epochs):
        loss_disc = []
        loss_gen = []
        for it in range(50):
            for i in range(1):
                real = read()
                noise = np.random.randn(13, 100)
                fake = gen.predict(noise)
                d_loss_real = disc.train_on_batch(real, np.ones([13]))[1]
                d_loss_fake = disc.train_on_batch(fake, np.zeros([13]))[1]
            if total_it % 80 == 0:
                plt.figure(figsize=(5, 2))
                real = read()
                noise = np.random.randn(5, 100)
                fake = gen.predict(noise)
                for obj_plot in [fake, real]:
                    fig = plt.figure(figsize=(15, 3))
                    for b in range(5):
                        disc_score = float(disc.predict(np.expand_dims(obj_plot[b], axis=0))[0])
                        plt.subplot(1, 5, b+1)
                        plt.title(str(round(disc_score, 3)))
                        plt.imshow(obj_plot[b]*.5+.5)
                    if obj_plot is fake:
                        plt.savefig(os.path.join('GAN', str(total_it).zfill(10) + '.jpg'), format='jpg', bbox_inches='tight')
                    # plt.show()
                    plt.close(fig)
            loss = 0
            y = np.ones([13, 1])
            for j in range(1):
                noise = np.random.randn(13, 100)
                loss += gan.train_on_batch(noise, y)[1]
            loss_disc.append((d_loss_real+d_loss_fake) / 2.)
            loss_gen.append(loss)
            total_it += 1
        print('Epoch', epoch)
        avg_loss_disc.append(np.mean(loss_disc))
        avg_loss_gen.append(np.mean(loss_gen))
        plt.plot(range(len(avg_loss_disc)), avg_loss_disc)
        plt.plot(range(len(avg_loss_gen)), avg_loss_gen)
        plt.legend(['discriminator loss', 'generator loss'])
        plt.show()

def main() -> None:
    gen, disc, gan = model(True)
    train(gen, disc, gan, epochs=50)


if __name__ == '__main__':
    main()