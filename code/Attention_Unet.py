import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Input, BatchNormalization, \
    Activation, Multiply


def conv_block(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation='relu'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def attention_block(inputs, skip_connection, filters, kernel_size=(1, 1), strides=(1, 1), padding='same',
                    activation='relu'):
    g = conv_block(inputs=skip_connection, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                   activation=activation)
    x = conv_block(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                   activation=activation)
    x = Multiply()([x, g])
    return x


def AttentionUNet(input_shape, num_classes=1, filters=64):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = conv_block(inputs=inputs, filters=filters, kernel_size=(3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(inputs=pool1, filters=filters * 2, kernel_size=(3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(inputs=pool2, filters=filters * 4, kernel_size=(3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(inputs=pool3, filters=filters * 8, kernel_size=(3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Center
    center = conv_block(inputs=pool4, filters=filters * 16, kernel_size=(3, 3))

    # Decoder
    up4 = UpSampling2D(size=(2, 2))(center)
    att4 = attention_block(inputs=up4, skip_connection=conv4, filters=filters * 8)
    concat4 = Concatenate()([up4, att4])
    conv5 = conv_block(inputs=concat4, filters=filters * 8, kernel_size=(3, 3))

    up3 = UpSampling2D(size=(2, 2))(conv5)
    att3 = attention_block(inputs=up3, skip_connection=conv3, filters=filters * 4)
    concat3 = Concatenate()([up3, att3])
    conv6 = conv_block(inputs=concat3, filters=filters * 4, kernel_size=(3, 3))

    up2 = UpSampling2D(size=(2, 2))(conv6)
    att2 = attention_block(inputs=up2, skip_connection=conv2, filters=filters * 2)
    concat2 = Concatenate()([up2, att2])
    conv7 = conv_block(inputs=concat2, filters=filters * 2, kernel_size=(3, 3))

    up1 = UpSampling2D(size=(2, 2))(conv7)
    att1 = attention_block(inputs=up1, skip_connection=conv1, filters=filters)
