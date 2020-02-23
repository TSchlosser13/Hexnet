'''****************************************************************************
 * GANs.py: GAN Models
 ******************************************************************************
 * v0.1 - 01.03.2019
 *
 * Copyright (c) 2019 Tobias Schlosser (tobias@tobias-schlosser.net)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ****************************************************************************'''


# https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py


import os

import numpy      as np
import tensorflow as tf

from matplotlib.pyplot       import imsave
from tensorflow.keras        import Input, Model, Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Embedding, Flatten, LeakyReLU, multiply, Reshape, UpSampling2D

from misc.misc import Hexnet_print


class ACGAN():
    def __init__(self, input_shape, classes):
        # Input shape
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.channels = input_shape[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = classes
        self.latent_dim = 100


    def compile(self):
        losses    = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        optimizer = tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        # Build and compile the discriminator
        self.discriminator, self.discriminator_summary = self.build_discriminator()
        self.discriminator.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator, self.generator_summary = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)


    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img), model


    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label]), model


    def fit(self, train_data, train_labels, batch_size=128, epochs=10000, tests_dir=None, run_title=None, images_to_sample_per_class=100, sample_rate=100, disable_training=False):
        # Load the dataset
        X_train, y_train = train_data, train_labels

        # Configure inputs
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(1, epochs + 1):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-9
            img_labels = y_train[idx]

            # Train the discriminator
            if not disable_training:
                d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            else:
                d_loss_real = self.discriminator.test_on_batch(imgs, [valid, img_labels])
                d_loss_fake = self.discriminator.test_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            if not disable_training:
                g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
            else:
                g_loss = self.combined.test_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            Hexnet_print(f'(epoch={epoch:{len(str(epochs))}}/{epochs}) [D loss={d_loss[0]:11.8f}, acc={100*d_loss[3]:6.2f}%, op_acc={100*d_loss[4]:6.2f}%] [G loss={g_loss[0]:11.8f}]')

            # If at save interval => save generated image samples
            if not epoch % sample_rate and tests_dir is not None:
                self.sample_images(epoch, tests_dir, run_title, images_to_sample_per_class)


    def evaluate(self, train_data, train_labels, batch_size=128, epochs=10, tests_dir=None, run_title=None, images_to_sample_per_class=100, sample_rate=1, disable_training=True):
        if not run_title is None:
            run_title = f'{run_title}_evaluation'

        self.fit(train_data, train_labels, batch_size, epochs, tests_dir, run_title, images_to_sample_per_class, sample_rate, disable_training)


    def sample_images(self, epoch, tests_dir, run_title, images_to_sample_per_class):
        Hexnet_print('Sampling images')

        r, c = images_to_sample_per_class, self.num_classes
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        tests_dir_samples = os.path.join(tests_dir, run_title)
        os.makedirs(tests_dir_samples, exist_ok=True)

        for image_counter, (image, label) in enumerate(zip(gen_imgs, sampled_labels)):
            image_filename = f'epoch{epoch}_label{label}_image{image_counter}.png'
            imsave(os.path.join(tests_dir_samples, image_filename), image)


    def summary(self):
        Hexnet_print('Generator')
        self.generator_summary.summary()

        Hexnet_print('Discriminator')
        self.discriminator_summary.summary()




def model_GAN_ACGAN_standalone(input_shape, classes):
    return ACGAN(input_shape, classes)


