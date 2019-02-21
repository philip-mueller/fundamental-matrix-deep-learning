from tensorflow.python.keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D, BatchNormalization, Lambda, \
    concatenate

from train.model.generator import _check_derived_feature_layers
from train.model.layers import reconstruction_layer, norm_layer
import tensorflow as tf


def regressor(bottleneck, derived_features_list, images,
              reconstruction_type, norm, derived_feature_layers, dropout,
              dense_sizes, channels, batchnorm, bottleneck_batchnorm, derived_features_batchnorm):
    """
    Applies the regressor to the input tensors bottleneck, derived_features_list and images,
    and returns the regressed fundamental matrix as tensor.

    :param bottleneck: Bottleneck input. Tensor (tf) of dimension (None, bottleneck_size).
    :param derived_features_list: List of required derived features.
        Depending on the parameter derived_feature_layers.
        Contain o to 4 elements which each represents a multi channeled image of different dimension
        ordered from large too small derived features.
        May be None if derived features inputs should not be used.
    :param images: Image pair input. Tensor (tf) of dimension (None, image_width, image_height, 2).
        May be None if image input should not be used.
    :param reconstruction_type: Type of reconstruction.
        Either  None (for simple reconstruction), 'reconstruct' or 'reconstruct_ext'.
    :param norm: Type of used norm. Either 'fro' or 'abs'.
    :param derived_feature_layers: Set or list of derived feature layer numbers which should be used in the regressor.
        E.g. for {0, 1, 2, 3} all derived feature layers would be used.
    :param dropout: Dropout probability of the dropout layer in the regressor.
    :param dense_sizes: 2 element list (or tuple) of positive integers
        containing the number of neurons in the 2 dense layers.
    :param channels: 5 element list (or tuple) of positive integers,
        defining the number of channels in the 4 downsampling blocks (first 4 elements)
            and in the feature conv block (last element)
        The number of channels for each convolutional layer is the same within each block.
    :param batchnorm: True if batchnorm layers should be used within the model.
    :param bottleneck_batchnorm: True if the bottleneck input should be processed by a batchnorm layer before
        being used in the regressor.
    :param derived_features_batchnorm: True if the derived features inputs should be processed by a
        batchnorm layer before being used in the regressor.
    :return: The regressed fundamental matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    x = regressor_downsample_block(bottleneck, derived_features_list, images,
                                   derived_feature_layers=derived_feature_layers, channels=channels,
                                   batchnorm=batchnorm, bottleneck_batchnorm=bottleneck_batchnorm,
                                   derived_features_batchnorm=derived_features_batchnorm)
    x = regressor_regression_block(x, reconstruction_type, norm, dropout, dense_sizes)
    return x


def regressor_regression_block(x, reconstruction_type, norm, dropout, dense_sizes):
    """
    Applies the regression block to the input tensor x.

    :param x: Input tensor (tf) of any flat dimension (None, input_neurons).
    :param reconstruction_type: Type of reconstruction.
        Either  None (for simple reconstruction), 'reconstruct' or 'reconstruct_ext'.
    :param norm:  Type of used norm. Either 'fro' or 'abs'.
    :param dropout:  Dropout probability of the dropout layer in the regressor.
    :param dense_sizes: 2 element list (or tuple) of positive integers
        containing the number of neurons in the 2 dense layers.
    :return: Regressed fundamental matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    assert len(dense_sizes) == 2

    x = Dropout(dropout, noise_shape=None, seed=None, name='regressor_reg_dropout')(x)
    x = Dense(dense_sizes[0], activation='relu', name='regressor_reg_fc1')(x)
    x = Dense(dense_sizes[1], activation='relu', name='regressor_reg_fc2')(x)

    x = reconstruction_layer(reconstruction_type)(x)

    x = norm_layer(norm)(x)
    return x


def regressor_conv_block(x, name, channels, downscale, batchnorm):
    """
    Applies a convolutional block to the input tensor x.

    :param x: Input tensor (tf), a possible multichanneled image.
        Dimension: (None, in_width, in_height, in_channels)
    :param name: Name prefix for all layers in this block.
    :param channels: Int defining the number of channels of both convolutional layers in the block.
    :param downscale: True if x should be downscaled by a max pooling layer.
    :param batchnorm; True if a batchnorm layer should be included in the block.
    :return: Output tensor (tf), the processed image.
        downscale == True => Dimension: (None, in_width/2, in_height/2, channels)
        downscale == False => Dimension: (None, in_width, in_height, channels)
    """
    x = Conv2D(channels, (3, 3), padding="same", strides=(1, 1), activation="relu", name=name+'_conv1')(x)
    x = Conv2D(channels, (3, 3), padding="same", strides=(1, 1), activation="relu", name=name+'_conv2')(x)
    if batchnorm:
        x = BatchNormalization(name=name+'_bn')(x)

    if downscale:
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=name+'_pool')(x)

    return x


def regressor_downsample_block(bottleneck, derived_features_list, images, derived_feature_layers, channels,
                               batchnorm, bottleneck_batchnorm, derived_features_batchnorm):
    """
    Applies the whole downsampling block, including up to 4 downsampling steps to the inputs.

    Starts from the image pair input and the highest derived features if given and downscales them
    step by step while adding more derived features. In the end the bottleneck input is concatenated.
    THe downsampling steps are only needed when the image input or the derived input features are given.

    :param bottleneck: Bottleneck input. Tensor (tf) of dimension (None, bottleneck_size).
    :param derived_features_list: List of required derived features.
        Depending on the parameter derived_feature_layers.
        Contain o to 4 elements which each represents a multi channeled image of different dimension
        ordered from large too small derived features.
        May be None if derived features inputs should not be used.
    :param images: Image pair input. Tensor (tf) of dimension (None, image_width, image_height, 2).
        May be None if image input should not be used.
    :param derived_feature_layers: Set or list of derived feature layer numbers which should be used in the regressor.
        E.g. for {0, 1, 2, 3} all derived feature layers would be used.
    :param channels: 5 element list (or tuple) of positive integers,
        defining the number of channels in the 4 downsampling blocks (first 4 elements)
            and in the feature conv block (last element)
        The number of channels for each convolutional layer is the same within each block.
    :param batchnorm: True if batchnorm layers should be used within the model.
    :param bottleneck_batchnorm: True if the bottleneck input should be processed by a batchnorm layer before
        being used in the regressor.
    :param derived_features_batchnorm: True if the derived features inputs should be processed by a
        batchnorm layer before being used in the regressor.
    :return: Downsampled and flattened output tensor (tf) containing processed infos of all the inputs.
        Dimension: (None, output_neurons).
    """
    assert len(channels) == 5

    bottleneck, derived_features_list = _normalize_features(bottleneck, derived_features_list,
                                                            bottleneck_batchnorm=bottleneck_batchnorm,
                                                            derived_features_batchnorm=derived_features_batchnorm)

    if images is not None and derived_features_list is not None:
        downsampled = _regressor_downsample_block_with_images_and_derived_features(images, derived_features_list,
                                                                                   derived_feature_layers, batchnorm,
                                                                                   channels)
    elif images is not None and derived_features_list is None:
        downsampled = _regressor_downsample_block_with_images(images, batchnorm, channels)

    elif images is None and derived_features_list is not None:
        downsampled = _regressor_downsample_block_with_derived_features(derived_features_list, derived_feature_layers,
                                                                        batchnorm, channels)

    if images is not None or derived_features_list is not None:
        downsampled = regressor_conv_block(downsampled, 'regressor_features',
                                           channels=channels[4], downscale=False, batchnorm=batchnorm)
        downsampled = Conv2D(1, (1, 1), padding="same", strides=(1, 1), activation="relu",
                             name='regressor_features_select')(downsampled)
        downsampled = Flatten(name='regressor_features_flatten')(downsampled)
        x = concatenate([downsampled, bottleneck], name='regressor_features_concat')
    else:
        x = bottleneck

    return x


def _normalize_features(bottleneck, derived_features_list, bottleneck_batchnorm, derived_features_batchnorm):
    """
    Applies batchnorms to the inputs of required.
    :param bottleneck: Bottleneck input.
    :param derived_features_list: Derived features inputs.
    :param bottleneck_batchnorm: True to use batchnorm for bottleneck input.
    :param derived_features_batchnorm: True to use batchnorm for all derived features inputs.
    :return: (bottleneck, derived_features_list). Normalized inputs (by applying batchnorms).
    """
    if bottleneck_batchnorm:
        bottleneck = BatchNormalization(name='bottleneck_bn')(bottleneck)

    if derived_features_batchnorm and derived_features_list is not None:
        derived_features_list = [BatchNormalization(name='derived_feature_%d_bn' % (i+1))(derived_feature)
                                 for (i, derived_feature) in enumerate(derived_features_list)]

    return bottleneck, derived_features_list


def _regressor_downsample_block_with_images(images, batchnorm, channels):
    x = images

    x = regressor_conv_block(x, 'regressor_down0', channels=channels[0], downscale=True, batchnorm=batchnorm)
    x = regressor_conv_block(x, 'regressor_down1', channels=channels[1], downscale=True, batchnorm=batchnorm)
    x = regressor_conv_block(x, 'regressor_down2', channels=channels[2], downscale=True, batchnorm=batchnorm)
    x = regressor_conv_block(x, 'regressor_down3', channels=channels[3], downscale=True, batchnorm=batchnorm)

    return x


def _regressor_downsample_block_with_derived_features(derived_features_list, derived_feature_layers, batchnorm, channels):
    assert len(derived_features_list) <= 4
    derived_feature_layers = _check_derived_feature_layers(derived_feature_layers)
    derived_features_list = _extend_derived_features_list(derived_features_list, derived_feature_layers)

    x = None

    # layer 0
    if 0 in derived_feature_layers:
        x = derived_features_list[0]
    if x is not None:
        x = regressor_conv_block(x, 'regressor_down0', channels=channels[0], downscale=True, batchnorm=batchnorm)

    # layer 2
    if 1 in derived_feature_layers:
        if x is None:
            x = derived_features_list[1]
        else:
            x = concatenate([x, derived_features_list[1]], name='regressor_down1_concat')
    if x is not None:
        x = regressor_conv_block(x, 'regressor_down1', channels=channels[1], downscale=True, batchnorm=batchnorm)

    # layer 2
    if 2 in derived_feature_layers:
        if x is None:
            x = derived_features_list[2]
        else:
            x = concatenate([x, derived_features_list[2]], name='regressor_down2_concat')
    if x is not None:
        x = regressor_conv_block(x, 'regressor_down2', channels=channels[2], downscale=True, batchnorm=batchnorm)

    # layer 3
    if 3 in derived_feature_layers:
        if x is None:
            x = derived_features_list[3]
        else:
            x = concatenate([x, derived_features_list[3]], name='regressor_down3_concat')
    if x is not None:
        x = regressor_conv_block(x, 'regressor_down3', channels=channels[3], downscale=True, batchnorm=batchnorm)

    assert x is not None
    return x


def _regressor_downsample_block_with_images_and_derived_features(images, derived_features_list, derived_feature_layers,
                                                                 batchnorm, channels):
    assert len(derived_features_list) <= 4
    derived_feature_layers = _check_derived_feature_layers(derived_feature_layers)
    derived_features_list = _extend_derived_features_list(derived_features_list, derived_feature_layers)
    x = images

    if 0 in derived_feature_layers:
        x = concatenate([x, derived_features_list[0]], name='regressor_down0_concat')
    x = regressor_conv_block(x, 'regressor_down0', channels=channels[0], downscale=True, batchnorm=batchnorm)

    if 1 in derived_feature_layers:
        x = concatenate([x, derived_features_list[1]], name='regressor_down1_concat')
    x = regressor_conv_block(x, 'regressor_down1', channels=channels[1], downscale=True, batchnorm=batchnorm)

    if 2 in derived_feature_layers:
        x = concatenate([x, derived_features_list[2]], name='regressor_down2_concat')
    x = regressor_conv_block(x, 'regressor_down2', channels=channels[2], downscale=True, batchnorm=batchnorm)

    if 3 in derived_feature_layers:
        x = concatenate([x, derived_features_list[3]], name='regressor_down3_concat')
    x = regressor_conv_block(x, 'regressor_down3', channels=channels[3], downscale=True, batchnorm=batchnorm)

    return x


def _extend_derived_features_list(derived_features_list, derived_feature_layers):
    """
    Extends the derived features list to always contain 4 elements representing the 4 derived feature layers.

    Layers which are not in derived_feature_layers will be None in the extended list
    :param derived_features_list: List of required derived features.
        Depending on the parameter derived_feature_layers.
        Contain o to 4 elements which each represents a multi channeled image of different dimension
        ordered from large too small derived features.
    :param derived_feature_layers: Set or list of derived feature layer numbers which should be used in the regressor.
        E.g. for {0, 1, 2, 3} all derived feature layers would be used.
    :return: List with 4 elements, representing the 4 derived features inputs. Some of the entries may be None.
    """
    extended_derived_features_list = []
    derived_features_list_index = 0
    for i in range(4):
        if i in derived_feature_layers:
            extended_derived_features_list.append(derived_features_list[derived_features_list_index])
            derived_features_list_index += 1
        else:
            extended_derived_features_list.append(None)
    return extended_derived_features_list
