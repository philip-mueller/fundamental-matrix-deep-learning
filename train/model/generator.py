from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, Flatten, Reshape, Dropout, MaxPooling2D, Conv2D, Conv2DTranspose, \
    concatenate


def generator(x, bottleneck_size, dropout, channels, skip_connections, weight_reg, activity_reg):
    """
    Applies the generator to the input tensor x and returns the output tensors of the generator.

    :param x: Input tensor (tf) of the image input to the generator.
        A single channeled image (batch).
        Dimension: (None, image_width, image_height, 1)
    :param bottleneck_size: Number of neurons in the bottleneck layer.
    :param dropout: Dropout probability of the layer before the bottleneck.
    :param channels:  4 element list (or tuple) of positive integers,
        defining the number of channels in the 4 downsampling blocks and the 4 upsampling block.
        The number of channels for each convolutional layer is the same within each down-/upsampling block.
        The number of channels in a downsampling block and the corresponding upsampling block are the same.
    :param skip_connections: True if skip connections should be used between down- and upsampling.
    :param weight_reg: float or None, specifies the L2 weight regularization strength of the bottlneck layer.
    :param activity_reg: float or None, specifies the L2 activity regularization strength of the bottlneck layer.
    :return: (x, bottleneck, downsampled, upsampled)
        - x: Generated image output tensor. Dimension: (None, image_width, image_height, 1)
        - bottleneck: Bottleneck output tensor. Dimension: (None, bottleneck_size)
        - downsampled: List of downsampled output tensors. Starting from not downsampled to highly downsampled.
                        Dimensions depend on the layer dimensions of the corresponding layers.
        - upsampled: List of upsampled output tensors. Starting from large too small (like downsampled).
                        Dimensions depend on the layer dimensions of the corresponding layers but they are the same
                        as the dimension of downsampled.
    """
    assert len(channels) == 4

    downsampled = []
    upsampled = []

    # --- downsampling ---
    x, down_out = downsampling_block(x, channels[0], 'generator_down0')
    downsampled.append(down_out)

    x, down_out = downsampling_block(x, channels[1], 'generator_down1')
    downsampled.append(down_out)

    x, down_out = downsampling_block(x, channels[2], 'generator_down2')
    downsampled.append(down_out)

    x, down_out = downsampling_block(x, channels[3], 'generator_down3')
    downsampled.append(down_out)

    # --- bottleneck ---
    x, bottleneck = bottleneck_block(x, bottleneck_size=bottleneck_size, dropout=dropout,
                                     weight_reg=weight_reg, activity_reg=activity_reg)

    # --- upsampling ---
    x = upsampling_block(x, downsampled[-1], channels[3], skip_connections, 'generator_up3')
    upsampled.append(x)

    x = upsampling_block(x, downsampled[-2], channels[2], skip_connections, 'generator_up2')
    upsampled.append(x)

    x = upsampling_block(x, downsampled[-3], channels[1], skip_connections, 'generator_up1')
    upsampled.append(x)

    x = upsampling_block(x, downsampled[-4], channels[0], skip_connections, 'generator_up0')
    upsampled.append(x)

    upsampled.reverse()  # upsampled is now in the same order as upsampled => from large to small (highly downsampled)

    # --- image output ---
    x = Conv2D(1, (1, 1), padding='same', strides=(1, 1), activation='relu', name='generator_out_conv')(x)

    return x, bottleneck, downsampled, upsampled


def downsampling_block(x, channels, name):
    """
    Applies a downsampling block to the input tensor x.

    :param x: Input tensor (tf), a possible multichanneled image.
        Dimension: (None, in_width, in_height, in_channels)
    :param channels: Int defining the number of channels of both convolutional layers in the block.
    :param name: Name prefix for all layers in this block.
    :return: (x, downsample_out)
        x: Output tensor (tf), the downsampled image. Dimension: (None, in_width/2, in_height/2, channels)
        downsample_out: Output of internal layer used for derived features and skip connections.
            tensor (tf) of dimension (None, in_width, in_height, channels)
    """
    # no batchnorm as only one sample is used
    x = Conv2D(channels, (3, 3), padding='same', strides=(1, 1), activation='relu', name=name + '_conv1')(x)
    x = Conv2D(channels, (3, 3), padding='same', strides=(1, 1), activation='relu', name=name + '_conv2')(x)
    downsample_out = x
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=name + '_pool')(x)

    return x, downsample_out


def upsampling_block(x, downsample_in, channels, skip_connection, name):
    """
    Applies an upsampling block to the input tensor x.

    :param x: Input tensor (tf), a possible multichanneled image.
        Dimension: (None, in_width, in_height, in_channels)
    :param downsample_in: Input from the skip connections.
        This is the downsample_out of the corresponding downsampling block of the same scale.
        Tensor (tf) of dimension (None, in_width*2, in_height*2, channels).
        Only relevant if skip_connections is True.
    :param channels: Int defining the number of channels of both convolutional layers
        and the transposed conv layer in the block.
    :param skip_connection: True if skip connections should be used.
        If skip connections are used, downsample_in is concatenated internally, if not downsample_in is ignored.
    :param name: Name prefix for all layers in this block.
    :return: Output tensor (tf), the upsampled image.
        Dimension: (None, in_width*2, in_height*2, channels)
    """
    x = Conv2DTranspose(channels, (2, 2), padding='same', strides=(2, 2), name=name + '_tconv')(x)

    if skip_connection:
        x = concatenate([downsample_in, x], name=name + '_concat')

    x = Conv2D(channels, (3, 3), padding='same', strides=(1, 1), activation='relu', name=name + '_conv1')(x)
    x = Conv2D(channels, (3, 3), padding='same', strides=(1, 1), activation='relu', name=name + '_conv2')(x)
    return x


def bottleneck_block(x, bottleneck_size, dropout, weight_reg, activity_reg):
    """
    Applies the bottleneck block to the input tensor x.

    :param x: Input tensor (tf), a possible multichanneled image.
        Dimension: (None, in_width, in_height, in_channels)
    :param bottleneck_size: Number of neurons in the bottleneck layer.
    :param dropout: Dropout used before the bottleneck layer.
    :param weight_reg: L2 weight regularization strength used in the bottleneck layer.
        Float or None if no regularization should be used.
    :param activity_reg: L2 activity regularization strength used in the bottleneck layer.
        Float or None if no regularization should be used.
    :return: (x, bottleneck)
        x: Output tensor (tf) of the bottleneck block. Same dimension as input: (None, in_width, in_height, in_channels)
        bottleneck: Tensor (tf) representing the values of the bottleneck layer. Dimension: (None, bottleneck_size)
    """
    bottleneck_in_shape = x.shape
    bottleneck_in_neurons = bottleneck_in_shape[1] * bottleneck_in_shape[2] * bottleneck_in_shape[3]
    weight_regularizer = l2(weight_reg) if weight_reg is not None else None
    activity_regularizer = l2(activity_reg) if activity_reg is not None else None

    x = Flatten(name='generator_bottleneck_flatten')(x)
    x = Dropout(dropout, noise_shape=None, seed=None, name='generator_bottleneck_dropout1')(x)
    x = Dense(bottleneck_size, activation='relu', name='generator_bottleneck_fc1',
              activity_regularizer=activity_regularizer, kernel_regularizer=weight_regularizer)(x)

    bottleneck = x

    x = Dense(bottleneck_in_neurons, activation='relu', name='generator_bottleneck_fc2')(x)
    x = Reshape((bottleneck_in_shape[1], bottleneck_in_shape[2], bottleneck_in_shape[3]),
                name='generator_bottleneck_unflatten')(x)

    return x, bottleneck


def extract_derived_features(downsampled_layers, upsampled_layers, derived_feature_layers):
    """
    Extracts the required derived features from the down- and upsampling layer outputs.

    Which derived features are extracted depends on derived_feature_layers.
    Each derived features output is created by concatenating the corresponding down- and upsampling layer.

    :param downsampled_layers: List of length 4 containing all outputs (downsample_out) of the downsamping blocks.
        Ordered from starting from the largest scale (not downsampled) to the smallest.
    :param upsampled_layers: List of length 4 containing all outputs (x) of the upsampling blocks.
        Ordered from starting from the largest scale (fully upsampled) to the smallest (once upsampled).
    :param derived_feature_layers: Set or list of size 0 to 4 containing the numbers of the derived features which are
        extracted. If it contains {0, 1, 2, 3} this would mean that all derived features are extracted.
    :return: List of length 0 to 4 containing the derived features. Ordered starting from derived_features_0
        if it is contained. Each element is a multichannel image tensor of different dimension
        (from large to small).
    """
    assert len(downsampled_layers) == 4 and len(upsampled_layers) == 4
    derived_feature_layers = _check_derived_feature_layers(derived_feature_layers)

    # combine (concatenate) downsampled and upsampled layers to feature layers
    derived_features_list = [concatenate([layer[0], layer[1]])
                             for layer in zip(downsampled_layers, upsampled_layers)]
    # extract only the derived features which are required (in derived_feature_layers)
    derived_features_list = [layer for (i, layer) in enumerate(derived_features_list) if (i in derived_feature_layers)]

    return derived_features_list


def _check_derived_feature_layers(derived_feature_layers):
    if isinstance(derived_feature_layers, int):
        derived_feature_layers = {derived_feature_layers}
    derived_feature_layers = set(derived_feature_layers)
    assert all([derived_feature_layers in [0, 1, 2, 3] for derived_feature_layers in derived_feature_layers])
    return derived_feature_layers
