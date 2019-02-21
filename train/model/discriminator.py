from tensorflow.python.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D


def discriminator(x, classifier_neurons=[1024, 512], channels=[64, 128, 256, 512], dropout=0.75):
    """
    Applies the discriminator to the input tensor x and returns the output tensor of the discriminator.

    :param x: Input tensor (tf) to the discriminator.
        A single channeled image (batch).
        Dimension: (None, image_width, image_height, 1)
    :param classifier_neurons: 2 element list (or tuple) of positive integers,
        defining the number of neurons in the two dense classifier layers.
    :param channels: 4 element list (or tuple) of positive integers,
        defining the number of channels in the 4 downsampling blocks.
        The number of channels for each convolutional layer is the same within each downsampling block.
    :param dropout: Float defining the dropout probability of the dropout layer.
    :return: Output tensor (tf) of the discriminator.
        Probability of the image beeing a true image.
        Dimension: (None, 1)
    """
    assert len(classifier_neurons) == 2
    assert len(channels) == 4

    # --- downsampling ---
    x = downsampling_block(x, channels[0], 'discriminator_down0')
    x = downsampling_block(x, channels[1], 'discriminator_down1')
    x = downsampling_block(x, channels[2], 'discriminator_down2')
    x = downsampling_block(x, channels[3], 'discriminator_down3')

    x = classifier_block(x, classifier_neurons, dropout)
    return x


def downsampling_block(x, channels, name):
    """
    Applies a downsampling block to the input tensor x.

    :param x: Input tensor (tf), a possible multichanneled image.
        Dimension: (None, in_width, in_height, in_channels)
    :param channels: Int defining the number of channels of both convolutional layers in the block.
    :param name: Name prefix for all layers in this block.
    :return: Output tensor (tf), the downsampled image.
        Dimension: (None, in_width/2, in_height/2, channels)
    """
    x = Conv2D(channels, (3, 3), padding='same', strides=(1, 1), activation='relu', name=name + '_conv1')(x)
    x = Conv2D(channels, (3, 3), padding='same', strides=(1, 1), activation='relu', name=name + '_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=name + '_pool')(x)
    return x


def classifier_block(x, neurons, dropout):
    """
    Applies the classifier block to the input tensor x.

    :param x: Input tensor (tf), a possible multichanneled image.
        Dimension: (None, in_width, in_height, in_channels
    :param neurons: 2 element list (or tuple) of positive integers,
        defining the number of neurons in the two dense classifier layers.
    :param dropout: Float defining the dropout probability of the dropout layer.
    :return: Output tensor (tf), single probability.
        Dimension: (None, 1)
    """
    assert len(neurons) == 2

    x = Flatten(name='discriminator_class_flatten')(x)
    x = Dropout(dropout, noise_shape=None, seed=None, name='discriminator_class_dropout')(x)
    x = Dense(neurons[0], activation='relu', name='discriminator_class_fc1')(x)
    x = Dense(neurons[1], activation='relu', name='discriminator_class_fc2')(x)
    x = Dense(1, activation='sigmoid', name='discriminator_class_sigmoid')(x)

    return x
