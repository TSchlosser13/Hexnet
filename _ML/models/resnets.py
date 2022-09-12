'''****************************************************************************
 * resnets.py: Square and Hexagonal ResNet Models
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


################################################################################
# References
################################################################################

'''
    - https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py

    - [a] ResNet v1: Deep Residual Learning for Image Recognition (2015), https://arxiv.org/abs/1512.03385
    - [b] ResNet v2: Identity Mappings in Deep Residual Networks  (2016), https://arxiv.org/abs/1603.05027
'''


################################################################################
# Imports
################################################################################

from tensorflow.keras              import Input, Model
from tensorflow.keras.layers       import Activation, add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten
from tensorflow.keras.regularizers import l2

from layers.layers import HAvgPool2D, HConv2D, SAvgPool2D, SConv2D




################################################################################
# ResNet depth
################################################################################

def resnet_get_depth(version=1, n=3):
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    return depth


################################################################################
# ResNet layer
################################################################################

def resnet_layer(
    inputs,
    num_filters         = 16,
    filter_size         =  1.0,
    kernel_size         =  3,
    strides             =  1,
    activation          = 'relu',
    batch_normalization = True,
    conv_first          = True,
    mode                = 'baseline'):

    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        filter_size (float): filter size factor (convolutional layers)
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """

    num_filters = int(filter_size * num_filters)

    if mode == 'baseline':
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))
    elif mode == 'S-ResNet':
        conv = SConv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='SAME')
    elif mode == 'H-ResNet':
        conv = HConv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='SAME')

    x = inputs

    if conv_first:
        x = conv(x)

        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)

    return x


################################################################################
# ResNet v1
################################################################################

def resnet_v1(
    input_shape,
    depth,
    stacks          =  3,
    filter_size     =  1.0,
    num_classes     = 10,
    mode            = 'baseline',
    disable_pooling = True):

    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        stacks (int): number of stacks
        filter_size (float): filter size factor (convolutional layers)
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition
    num_filters    = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x      = resnet_layer(inputs, filter_size=filter_size, mode=mode)

    # Instantiate the stack of residual units
    for stack in range(stacks):
        if stack == stacks - 1:
            filter_size = 1.0

        for res_block in range(num_res_blocks):
            strides = 1

            if stack > 0 and res_block == 0: # first layer but not first stack
                strides = 2 # downsample

            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             filter_size=filter_size,
                             strides=strides,
                             mode=mode)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             filter_size=filter_size,
                             activation=None,
                             mode=mode)

            if stack > 0 and res_block == 0: # first layer but not first stack
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 filter_size=filter_size,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 mode=mode)

            x = add([x, y])
            x = Activation('relu')(x)

        num_filters *= 2

    # Add classifier on top
    # v1 does not use BN after last shortcut connection-ReLU

    if not disable_pooling:
        if mode == 'baseline':
            x = AveragePooling2D(pool_size=8)(x)
        elif mode == 'S-ResNet':
            x = SAvgPool2D(pool_size=(2, 2), padding='SAME')(x)
        elif mode == 'H-ResNet':
            x = HAvgPool2D(pool_size=(3, 3), padding='SAME')(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)

    return model


################################################################################
# ResNet v2
################################################################################

def resnet_v2(
    input_shape,
    depth,
    stacks          =  3,
    filter_size     =  1.0,
    num_classes     = 10,
    mode            = 'baseline',
    disable_pooling = True):

    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        stacks (int): number of stacks
        filter_size (float): filter size factor (convolutional layers)
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    # Start model definitio
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     filter_size=filter_size,
                     conv_first=True,
                     mode=mode)

    # Instantiate the stack of residual units
    for stage in range(stacks):
        if stack == stacks - 1:
            filter_size = 1.0

        for res_block in range(num_res_blocks):
            activation          = 'relu'
            batch_normalization = True
            strides             = 1

            if stage == 0:
                num_filters_out = num_filters_in * 4

                if res_block == 0: # first layer and first stage
                    activation          = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2

                if res_block == 0: # first layer but not first stage
                    strides = 2 # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             filter_size=filter_size,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             mode=mode)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             filter_size=filter_size,
                             conv_first=False,
                             mode=mode)

            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             filter_size=filter_size,
                             kernel_size=1,
                             conv_first=False,
                             mode=mode)

            if res_block == 0:
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 filter_size=filter_size,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 mode=mode)

            x = add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if not disable_pooling:
        if mode == 'baseline':
            x = AveragePooling2D(pool_size=8)(x)
        elif mode == 'S-ResNet':
            x = SAvgPool2D(pool_size=(2, 2), padding='SAME')(x)
        elif mode == 'H-ResNet':
            x = HAvgPool2D(pool_size=(3, 3), padding='SAME')(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)

    return model




################################################################################
# Square and Hexagonal ResNet Models
################################################################################


def model_ResNet_v1(input_shape, classes, stacks=3, n=3, filter_size=1.0):
    depth = resnet_get_depth(version=1, n=n)

    return resnet_v1(input_shape, depth, stacks, filter_size, classes, mode='baseline', disable_pooling=True)


def model_ResNet_v2(input_shape, classes, stacks=3, n=2, filter_size=1.0):
    depth = resnet_get_depth(version=2, n=n)

    return resnet_v2(input_shape, depth, stacks, filter_size, classes, mode='baseline', disable_pooling=True)


def model_SResNet_v1(input_shape, classes, stacks=3, n=3, filter_size=1.0):
    depth = resnet_get_depth(version=1, n=n)

    return resnet_v1(input_shape, depth, stacks, filter_size, classes, mode='S-ResNet', disable_pooling=True)


def model_SResNet_v2(input_shape, classes, stacks=3, n=2, filter_size=1.0):
    depth = resnet_get_depth(version=2, n=n)

    return resnet_v2(input_shape, depth, stacks, filter_size, classes, mode='S-ResNet', disable_pooling=True)


def model_HResNet_v1(input_shape, classes, stacks=3, n=3, filter_size=1.0):
    depth = resnet_get_depth(version=1, n=n)

    return resnet_v1(input_shape, depth, stacks, filter_size, classes, mode='H-ResNet', disable_pooling=True)


def model_HResNet_v2(input_shape, classes, stacks=3, n=2, filter_size=1.0):
    depth = resnet_get_depth(version=2, n=n)

    return resnet_v2(input_shape, depth, stacks, filter_size, classes, mode='H-ResNet', disable_pooling=True)


