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
# https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py


import os

import numpy      as np
import tensorflow as tf

from matplotlib.pyplot       import imsave
from tensorflow.keras        import Input, Model, Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout, Embedding, Flatten, LeakyReLU, multiply, Reshape, UpSampling2D
from tqdm                    import tqdm

from layers.layers      import HConv2D, HConv2DTranspose, HSampling2D, SConv2D, SConv2DTranspose, SSampling2D
from misc.misc          import Hexnet_print, normalize_array, print_newline
from misc.visualization import visualize_hexarray


class ACGAN():
    def __init__(self, input_shape, classes, latent_dim=100, mode='baseline'):
        self.input_shape = input_shape
        self.channels    = self.input_shape[2]
        self.classes     = classes
        self.latent_dim  = latent_dim
        self.mode        = mode


    def build_generator(self):
        model = Sequential()


        # https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py
        # https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py

        model.add(Dense(units = 128 * 8 * 8, activation = 'relu', input_dim = self.latent_dim))
        model.add(Reshape(target_shape = (8, 8, 128)))

        if self.mode == 'baseline':
            # model.add(UpSampling2D())
            # model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
            model.add(Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same'))
        elif self.mode == 'S-ACGAN':
            # model.add(SSampling2D())
            # model.add(SConv2D(filters=128, kernel_size=3, padding='SAME'))
            model.add(SConv2DTranspose(filters=128, kernel_size=3, strides=2, padding='SAME'))
        elif self.mode == 'H-ACGAN':
            # model.add(HSampling2D())
            # model.add(HConv2D(filters=128, kernel_size=3, padding='SAME'))
            model.add(HConv2DTranspose(filters=128, kernel_size=3, strides=2, padding='SAME'))

        model.add(Activation('relu'))
        model.add(BatchNormalization())

        if self.mode == 'baseline':
            # model.add(UpSampling2D())
            # model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
            model.add(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'))
        elif self.mode == 'S-ACGAN':
            # model.add(SSampling2D())
            # model.add(SConv2D(filters=64, kernel_size=3, padding='SAME'))
            model.add(SConv2DTranspose(filters=64, kernel_size=3, strides=2, padding='SAME'))
        elif self.mode == 'H-ACGAN':
            # model.add(HSampling2D())
            # model.add(HConv2D(filters=64, kernel_size=3, padding='SAME'))
            model.add(HConv2DTranspose(filters=64, kernel_size=3, strides=2, padding='SAME'))

        model.add(Activation('relu'))
        model.add(BatchNormalization())

        if self.mode == 'baseline':
            model.add(Conv2D(filters=self.channels, kernel_size=3, padding='same'))
        elif self.mode == 'S-ACGAN':
            model.add(SConv2D(filters=self.channels, kernel_size=3, padding='SAME'))
        elif self.mode == 'H-ACGAN':
            model.add(HConv2D(filters=self.channels, kernel_size=3, padding='SAME'))

        model.add(Activation('tanh'))


        noise           = Input(shape=(self.latent_dim,))
        label           = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.classes, self.latent_dim)(label))
        model_input     = multiply([noise, label_embedding])
        img             = model(model_input)

        return (Model([noise, label], img), model)


    def build_discriminator(self):
        model = Sequential()


        # https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py

        if self.mode == 'baseline':
            model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', input_shape=self.input_shape))
        elif self.mode == 'S-ACGAN':
            model.add(SConv2D(filters=32, kernel_size=3, strides=2, padding='SAME', input_shape=self.input_shape))
        elif self.mode == 'H-ACGAN':
            model.add(HConv2D(filters=32, kernel_size=3, strides=2, padding='SAME', input_shape=self.input_shape))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))

        if self.mode == 'baseline':
            model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        elif self.mode == 'S-ACGAN':
            model.add(SConv2D(filters=64, kernel_size=3, padding='SAME'))
        elif self.mode == 'H-ACGAN':
            model.add(HConv2D(filters=64, kernel_size=3, padding='SAME'))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))

        if self.mode == 'baseline':
            model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding='same'))
        elif self.mode == 'S-ACGAN':
            model.add(SConv2D(filters=128, kernel_size=3, strides=2, padding='SAME'))
        elif self.mode == 'H-ACGAN':
            model.add(HConv2D(filters=128, kernel_size=3, strides=2, padding='SAME'))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))

        if self.mode == 'baseline':
            model.add(Conv2D(filters=256, kernel_size=3, padding='same'))
        elif self.mode == 'S-ACGAN':
            model.add(SConv2D(filters=256, kernel_size=3, padding='SAME'))
        elif self.mode == 'H-ACGAN':
            model.add(HConv2D(filters=256, kernel_size=3, padding='SAME'))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(rate=0.3))
        model.add(Flatten())


        img      = Input(shape=self.input_shape)
        features = model(img)
        validity = Dense(units=1, activation='sigmoid')(features)
        label    = Dense(units=self.classes, activation='softmax')(features)

        return (Model(img, [validity, label]), model)


    def compile(self):
        losses    = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        optimizer = tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        (self.discriminator, self.discriminator_for_summary) = self.build_discriminator()
        self.discriminator.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])

        (self.generator, self.generator_for_summary) = self.build_generator()

        # The generator takes noise and the target label as input and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img   = self.generator([noise, label])

        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses, optimizer=optimizer)


    def sample_images(self, epoch, visualize_hexagonal, output_dir, run_title, images_to_sample_per_class):
        r, c           = images_to_sample_per_class, self.classes
        noise          = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.asarray([num for _ in range(r) for num in range(c)])
        gen_imgs       = self.generator.predict([noise, sampled_labels])
        gen_imgs       = normalize_array(gen_imgs)

        output_dir_samples = os.path.join(output_dir, run_title)
        os.makedirs(output_dir_samples, exist_ok=True)

        for image_counter, (image, label) in enumerate(zip(gen_imgs, sampled_labels)):
            image_filename = f'epoch{epoch}_label{label}_image{image_counter}.png'

            if not visualize_hexagonal:
                imsave(os.path.join(output_dir_samples, image_filename), image)
            else:
                visualize_hexarray(image, os.path.join(output_dir_samples, image_filename))


    def fit(
        self,
        train_data,
        train_labels,
        batch_size                 = 100,
        epochs                     = 100,
        visualize_hexagonal        = False,
        output_dir                 = None,
        run_title                  = None,
        images_to_sample_per_class =  10,
        disable_training           = False):

        # Configure inputs
        train_labels = np.reshape(train_labels, newshape = (-1, 1))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake  = np.zeros((batch_size, 1))

        batches = int(train_data.shape[0] / batch_size)


        for epoch in range(1, epochs + 1):
            for batch in tqdm(range(1, batches + 1)):

                ################################################################
                # Train the discriminator
                ################################################################

                # Select a random batch of images
                idx  = np.random.randint(0, train_data.shape[0], batch_size)
                imgs = train_data[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # The labels of the digits that the generator tries to create an image representation of
                sampled_labels = np.random.randint(0, self.classes, (batch_size, 1))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, sampled_labels])

                # Image labels
                img_labels = train_labels[idx]

                if not disable_training:
                    d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
                else:
                    d_loss_real = self.discriminator.test_on_batch(imgs, [valid, img_labels])
                    d_loss_fake = self.discriminator.test_on_batch(gen_imgs, [fake, sampled_labels])

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


                ################################################################
                # Train the generator
                ################################################################

                if not disable_training:
                    g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
                else:
                    g_loss = self.combined.test_on_batch([noise, sampled_labels], [valid, sampled_labels])


            Hexnet_print(f'(epoch={epoch:{len(str(epochs))}}/{epochs}) [D loss={d_loss[0]:11.8f}, acc={100*d_loss[3]:6.2f}%, op_acc={100*d_loss[4]:6.2f}%] [G loss={g_loss[0]:11.8f}]')

            if output_dir is not None:
                self.sample_images(epoch, visualize_hexagonal, output_dir, run_title, images_to_sample_per_class)


    def evaluate(
        self,
        train_data,
        train_labels,
        batch_size                 = 100,
        epochs                     =  10,
        visualize_hexagonal        = False,
        output_dir                 = None,
        run_title                  = None,
        images_to_sample_per_class =  10,
        disable_training           = True):

        if run_title is not None:
            run_title = f'{run_title}_evaluation'

        self.fit(
            train_data,
            train_labels,
            batch_size,
            epochs,
            visualize_hexagonal,
            output_dir,
            run_title,
            images_to_sample_per_class,
            disable_training)


    def summary(self):
        Hexnet_print('Generator')
        self.generator_for_summary.summary()

        print_newline()
        Hexnet_print('Discriminator')
        self.discriminator_for_summary.summary()




def model_GAN_ACGAN_standalone(input_shape, classes):
    return ACGAN(input_shape, classes, mode='baseline')


def model_GAN_SACGAN_standalone(input_shape, classes):
    return ACGAN(input_shape, classes, mode='S-ACGAN')


def model_GAN_HACGAN_standalone(input_shape, classes):
    return ACGAN(input_shape, classes, mode='H-ACGAN')


