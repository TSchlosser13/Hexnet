'''****************************************************************************
 * autoencoders.py: Autoencoder Models
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


# https://github.com/keras-team/keras/blob/master/examples/mnist_swwae.py


'''Trains a stacked what-where autoencoder built on residual blocks on the
MNIST dataset. It exemplifies two influential methods that have been developed
in the past few years.

The first is the idea of properly 'unpooling.' During any max pool, the
exact location (the 'where') of the maximal value in a pooled receptive field
is lost, however it can be very useful in the overall reconstruction of an
input image. Therefore, if the 'where' is handed from the encoder
to the corresponding decoder layer, features being decoded can be 'placed' in
the right location, allowing for reconstructions of much higher fidelity.

# References

- Visualizing and Understanding Convolutional Networks
  Matthew D Zeiler, Rob Fergus
  https://arxiv.org/abs/1311.2901v3
- Stacked What-Where Auto-encoders
  Junbo Zhao, Michael Mathieu, Ross Goroshin, Yann LeCun
  https://arxiv.org/abs/1506.02351v8

The second idea exploited here is that of residual learning. Residual blocks
ease the training process by allowing skip connections that give the network
the ability to be as linear (or non-linear) as the data sees fit.  This allows
for much deep networks to be easily trained. The residual element seems to
be advantageous in the context of this example as it allows a nice symmetry
between the encoder and decoder. Normally, in the decoder, the final
projection to the space where the image is reconstructed is linear, however
this does not have to be the case for a residual block as the degree to which
its output is linear or non-linear is determined by the data it is fed.
However, in order to cap the reconstruction in this example, a hard softmax is
applied as a bias because we know the MNIST digits are mapped to [0, 1].

# References
- Deep Residual Learning for Image Recognition
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  https://arxiv.org/abs/1512.03385v1
- Identity Mappings in Deep Residual Networks
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  https://arxiv.org/abs/1603.05027v3
'''


import numpy as np

from tensorflow.keras         import Input, Model
from tensorflow.keras.backend import gradients, sum
from tensorflow.keras.layers  import Activation, add, BatchNormalization, Conv2D, ELU, Lambda, MaxPooling2D, multiply, UpSampling2D


def convresblock(x, nfeats=8, ksize=3, nskipped=2, elu=True):
    """The proposed residual block from [4].

    Running with elu=True will use ELU nonlinearity and running with
    elu=False will use BatchNorm + RELU nonlinearity.  While ELU's are fast
    due to the fact they do not suffer from BatchNorm overhead, they may
    overfit because they do not offer the stochastic element of the batch
    formation process of BatchNorm, which acts as a good regularizer.

    # Arguments
        x: 4D tensor, the tensor to feed through the block
        nfeats: Integer, number of feature maps for conv layers.
        ksize: Integer, width and height of conv kernels in first convolution.
        nskipped: Integer, number of conv layers for the residual function.
        elu: Boolean, whether to use ELU or BN+RELU.

    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)`

    # Output shape
        4D tensor with shape:
        `(batch, filters, rows, cols)`
    """
    y0 = Conv2D(nfeats, ksize, padding='same')(x)
    y = y0
    for i in range(nskipped):
        if elu:
            y = ELU()(y)
        else:
            y = BatchNormalization(axis=1)(y)
            y = Activation('relu')(y)
        y = Conv2D(nfeats, 1, padding='same')(y)
    return add([y0, y])


def getwhere(x):
    ''' Calculate the 'where' mask that contains switches indicating which
    index contained the max value when MaxPool2D was applied.  Using the
    gradient of the sum is a nice trick to keep everything high level.'''
    y_prepool, y_postpool = x
    return gradients(sum(y_postpool), y_prepool)


def autoencoder_SWWAE(input_shape, pool_size = 2, nfeats = [8, 16, 32, 64, 128], ksize = 3, nlayers = 5):
    pool_sizes = np.asarray([1, 1, 1, 1, 1]) * pool_size
    nfeats_all = [input_shape[2]] + nfeats

    img_input = Input(shape=input_shape)
    wheres    = [None] * nlayers
    y         = img_input

    for i in range(nlayers):
        y_prepool = convresblock(y, nfeats=nfeats_all[i + 1], ksize=ksize)
        y         = MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i]))(y_prepool)
        wheres[i] = Lambda(getwhere, output_shape=lambda x: x[0])([y_prepool, y])[0]

    for i in range(nlayers):
        ind = nlayers - 1 - i
        y   = UpSampling2D(size=(pool_sizes[ind], pool_sizes[ind]))(y)
        y   = multiply([y, wheres[ind]])
        y   = convresblock(y, nfeats=nfeats_all[ind], ksize=ksize)

    y     = Activation('hard_sigmoid')(y)
    model = Model(img_input, y)

    return model




def model_autoencoder_SWWAE(input_shape, number_of_layers = 5, number_of_features = [8, 16, 32, 64, 128], kernel_size = 3, pool_size = 2):
    return autoencoder_SWWAE(input_shape, pool_size, nfeats=number_of_features, ksize=kernel_size, nlayers=number_of_layers)

