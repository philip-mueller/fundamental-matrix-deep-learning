from tensorflow.python.keras import Input, Model

from tensorflow.python.keras._impl.keras.engine.topology import Network
from tensorflow.python.keras.layers import concatenate

from train.model import discriminator
from train.model.generator import generator, extract_derived_features
from train.model.regressor import regressor


def build_generator_with_regressor_models(img_size, params):
    """
    Builds the keras models for the generator combined with the regressor.

    Multiple models are build:
     - generator_with_regressor_model: the combined model,
         which has both images as input and the generated image as well as the regressed fundamental matrix as output
     - generator_model: the generator only model, which has the first image as input and the generated image as output
     - generator_with_output_model: the generator model with regresison outputs,
         which has the first image as input and the generated image as well as the features needed for the regressor
         as outputs.
     - regressor_model: the regressor alone, which has regression inputs as input and the fundamental matrix as output

    :param img_size: (image_width, image_height), defining the size of the input and output image.
    :param params: Dictionary of hyperparameters for the generator and the regressor.
    :return: generator_with_regressor_model, generator_model, generator_with_output_model, regressor_model
    """
    input_shape_single = (img_size[0], img_size[1], 1)

    # params
    use_images = params['use_images']
    use_derived_features = params['derived_feature_layers'] is not None

    # --- Image input(s) ---
    img_A_input = Input(shape=input_shape_single, name='image_A_input')
    if use_images:
        img_B_input = Input(shape=input_shape_single, name='image_B_input')
        img_pair_input = concatenate([img_A_input, img_B_input])

    # --- Generator ---
    generator_model, generator_with_output_model = build_generator_model(img_size, params)

    if use_derived_features:
        generator_output, bottleneck, *derived_features_list = generator_with_output_model(img_A_input)
        # get shapes for all derived features: batch dimension does not matter -> [1:]
        derived_features_shapes = [derived_feature.shape[1:] for derived_feature in derived_features_list]
    else:
        generator_output, bottleneck = generator_with_output_model(img_A_input)
        derived_features_shapes = None

    # --- Regressor ---
    regressor_model = build_regressor_model(img_size, derived_features_shapes, params)

    if use_images and use_derived_features:
        regressor_output = regressor_model([bottleneck, *derived_features_list, img_pair_input])
    elif not use_images and use_derived_features:
        regressor_output = regressor_model([bottleneck, *derived_features_list])
    elif use_images and not use_derived_features:
        regressor_output = regressor_model([bottleneck, img_pair_input])
    else:
        regressor_output = regressor_model(bottleneck)

    # --- Models ---
    if use_images:
        generator_with_regressor_model = Model(inputs=[img_A_input, img_B_input],
                                               outputs=[generator_output, regressor_output],
                                               name='generator_with_regressor_model')
    else:
        generator_with_regressor_model = Model(inputs=[img_A_input],
                                               outputs=[generator_output, regressor_output],
                                               name='generator_with_regressor_model')

    return generator_with_regressor_model, generator_model, generator_with_output_model, regressor_model


def build_generator_model(img_size, params):
    """
    Builds the keras models for the generator.

    Two models are build, the generator only with the generated image as output and a model which also
    has the features for the regressor (bottleneck and derived feature layers) as output.

    :param img_size: (image_width, image_height), defining the size of the input and output image.
    :param params: Dictionary of hyperparameters for the generator.
    :return: generator_model, generator_with_output_model.
        Both are keras models with input dimension (None, image_width, image_height, 1).
        generator_model has one output with dimension (None, image_width, image_height, 1).
        generator_with_output_model has two or more outputs,
         - the first is the generated image with dimension (None, image_width, image_height, 1)
         - the second is the bottleneck with dimension (None, bottleneck_size)
         - the other optional outputs depend on 'derived_feature_layers' and have the dimensions of the
           downsampled feature layers.
    """
    input_shape_single = (img_size[0], img_size[1], 1)

    # params
    bottleneck_size = params['bottleneck_size']
    channels = params['generator_channels']
    dropout = params['generator_dropout']
    derived_feature_layers = params['derived_feature_layers']
    skip_connections = params['generator_skip_connections']
    weight_reg = params['generator_bottleneck_weight_reg']
    activity_reg = params['generator_bottleneck_activity_reg']

    # input
    generator_input = Input(shape=input_shape_single, name='generator_input')

    # outputs
    generator_output, bottleneck, downsampled, upsampled = generator(generator_input, bottleneck_size=bottleneck_size,
                                                                     dropout=dropout, channels=channels,
                                                                     skip_connections=skip_connections,
                                                                     weight_reg=weight_reg, activity_reg=activity_reg)
    if derived_feature_layers is not None:
        derived_features_list = extract_derived_features(downsampled, upsampled, derived_feature_layers)
    else:
        derived_features_list, active_derived_features_list = None, None

    # models
    generator_model = Model(inputs=generator_input, outputs=[generator_output], name='generator_model')
    if derived_feature_layers is not None:
        generator_with_output_model = Model(inputs=generator_input,
                                            outputs=[generator_output, bottleneck, *derived_features_list],
                                            name='generator_with_output_model')
    else:
        generator_with_output_model = Model(inputs=generator_input,
                                            outputs=[generator_output, bottleneck],
                                            name='generator_with_output_model')

    return generator_model, generator_with_output_model


def build_regressor_model(img_size, derived_features_shapes, params):
    """
    Builds the keras model for the regressor.

    The input(s) of the model depend on the hyperparameters 'use_images' and 'derived_feature_layers'.
    It always has the bottleneck as input, optionally it also uses the derived feature layers (one or more of them)
    when 'derived_feature_layers' is not None
    and optionally also the image pair when 'use_images' = True.
    The only regressor output is the fundamental matrix.

    :param img_size: (image_width, image_height), defining the size of the input image.
    :param derived_features_shapes:
    :param params: Dictionary of hyperparameters for the regressor.
    :return: regressor_discriminator_model.
        Keras model with the possible inputs:
         - bottleneck, dimension: (None, bottleneck_size)
         - derived_feature_layers (optional, multiple inputs), dimension: dimensions of derived feature layers
         - image_pair (optional), dimension: (None, image_width, image_height, 2)
        Output dimension: (None, 3, 3)
    """

    # params
    use_images = params['use_images']
    derived_feature_layers = params['derived_feature_layers']
    use_derived_features = derived_feature_layers is not None
    bottleneck_size = params['bottleneck_size']
    reconstruction_type = params['reconstruction_type']
    norm = params['norm']
    dense_sizes = params['regressor_dense_neurons']
    channels = params['regressor_channels']
    dropout = params['regressor_dropout']
    batchnorm = params['regressor_batchnorm']
    bottleneck_batchnorm = params['regressor_bottleneck_batchnorm']
    derived_features_batchnorm = params['regressor_derived_features_batchnorm']

    # inputs
    bottleneck_input = Input(shape=[bottleneck_size], name='regressor_bottleneck_input')

    if use_images:
        images_input = Input(shape=(img_size[0], img_size[1], 2), name='regressor_image_pair_input')
    else:
        images_input = None

    if use_derived_features:
        derived_features_input_list = [Input(shape=derived_features_shape,
                                             name='regressor_derived_features_%d_input' % i)
                                       for (i, derived_features_shape) in enumerate(derived_features_shapes)]
    else:
        derived_features_input_list = None

    # output
    regressor_output = regressor(bottleneck_input, derived_features_input_list, images_input,
                                 reconstruction_type=reconstruction_type, norm=norm,
                                 derived_feature_layers=derived_feature_layers,
                                 dropout=dropout, dense_sizes=dense_sizes, channels=channels, batchnorm=batchnorm,
                                 bottleneck_batchnorm=bottleneck_batchnorm,
                                 derived_features_batchnorm=derived_features_batchnorm)

    # models
    if use_images and use_derived_features:
        regressor_model = Model(inputs=[bottleneck_input, *derived_features_input_list, images_input],
                                outputs=[regressor_output],
                                name='regressor_model')
    elif not use_images and use_derived_features:
        regressor_model = Model(inputs=[bottleneck_input, *derived_features_input_list],
                                outputs=[regressor_output],
                                name='regressor_model')
    elif use_images and not use_derived_features:
        regressor_model = Model(inputs=[bottleneck_input, images_input],
                                outputs=[regressor_output],
                                name='regressor_model')
    else:
        regressor_model = Model(inputs=[bottleneck_input],
                                outputs=[regressor_output],
                                name='regressor_model')

    return regressor_model


def build_discriminator_models(img_size, params):
    """
    Builds keras models for the discriminator.

    Two models are created: the normal discriminator model and a frozen discriminator model, which is just a wrapper
    around the normal discriminator. Both share the same weights, but the weights of the frozen discriminator
    can be frozen independently.

    :param img_size: (image_width, image_height), defining the size of the input image.
    :param params: Dictionary of hyperparameters for the discriminator.
    :return: discriminator_model, frozen_discriminator_model.
        Both are keras models with input dimension (None, image_width, image_height, 1) and
        output dimension (None, 1)
    """
    input_shape = (img_size[0], img_size[1], 1)

    # params
    classifier_neurons = params['discriminator_classifier_neurons']
    channels = params['discriminator_channels']
    dropout = params['discriminator_dropout']

    # input
    input = Input(shape=input_shape, name='discriminator_input')

    # output
    output = discriminator.discriminator(input, classifier_neurons, channels, dropout)

    # models
    discriminator_model = Model(inputs=input, outputs=[output], name='discriminator_model')
    frozen_discriminator_model = Network(
        input,
        output,
        name='frozen_discriminator_model'
    )
    return discriminator_model, frozen_discriminator_model
