from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
import numpy as np
import os
from matplotlib import pyplot as plt

from train.utils.metrics import image_mse
from train.model.model_builders import build_discriminator_models, build_generator_with_regressor_models
from train.utils.params import Params

TMP_WEIGHTS_FILE_PREFIX = 'tmp_weights'


class FMatrixGanModel:
    """
    Defines the complete model with generator, regressor and discriminator.
    This includes the low level training and prediction methods for this model, like the GAN training.
    """
    def __init__(self, params, model_folder, img_size):
        """
        Inits the model.

        :param params: Hyperparameters
        :param model_folder: Folder path, in which all results and temporary data of the model is stored.
        :param img_size: (image_width, image_height), defining the size of the input images.
        """
        if not isinstance(params, Params):
            params = Params(params)
        self.params = params

        self.model_folder = model_folder

        # inputs
        input_shape = (img_size[0], img_size[1], 1)
        img_A, img_B = Input(shape=input_shape), Input(shape=input_shape)

        # --- build models
        discriminator_model, frozen_discriminator_model = build_discriminator_models(img_size, params)

        generator_with_regressor_model, generator_model, generator_with_output_model, regressor_model = \
            build_generator_with_regressor_models(img_size, params)

        # --- models
        self.discriminator = discriminator_model
        self.regressor = regressor_model
        self.generator = generator_model
        self.generator_with_output = generator_with_output_model
        self.generator_with_regressor = generator_with_regressor_model

        # model: GAN without regressor and without output
        fake_B = generator_model(img_A)
        gan_out = frozen_discriminator_model(fake_B)
        self.gan = Model(inputs=img_A, outputs=gan_out)

        # model: GAN with regressor
        if params['use_images']:
            fake_B, fmatrix = generator_with_regressor_model([img_A, img_B])
            gan_out = frozen_discriminator_model(fake_B)
            self.gan_with_regressor = Model(inputs=[img_A, img_B], outputs=[gan_out, fmatrix])
        else:
            fake_B, fmatrix = generator_with_regressor_model(img_A)
            gan_out = frozen_discriminator_model(fake_B)
            self.gan_with_regressor = Model(inputs=img_A, outputs=[gan_out, fmatrix])

        # --- compile models
        self.discriminator.compile(loss='binary_crossentropy',
                              optimizer=Adam(lr=params['lr_D'], beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                              metrics=['accuracy'])
        self.regressor.compile(loss='mean_squared_error',
                               optimizer=Adam(lr=params['lr_R'], beta_1=0.9, beta_2=0.999, epsilon=1e-08))

        # generators do not need to be compiled as they are compiled within the GANs

        if params['freeze_discriminator']:
            frozen_discriminator_model.trainable = False

        self.gan.compile(loss='binary_crossentropy',
                         optimizer=Adam(lr=params['lr_G'], beta_1=0.9, beta_2=0.999, epsilon=1e-08))

        loss_weights = params['generator_loss_weights']
        assert len(loss_weights) == 2
        self.gan_with_regressor.compile(loss=['binary_crossentropy', 'mean_squared_error'], loss_weights=loss_weights,
                                        optimizer=Adam(lr=params['lr_G'], beta_1=0.9, beta_2=0.999, epsilon=1e-08))

        self.__models_with_weights = [self.generator, self.regressor, self.discriminator]

    def generate_img(self, img_A):
        """
        Generates an image from img_A using the generator and its current weights.
        :param img_A: Input image to the generator. Dimension: (img_width, img_height)
        :return: The generated image
        """
        img_A = _img_to_img_batch(img_A)
        img_B = self.generator.predict(img_A)
        return img_B[0]  # only 1 sample in batch

    def generate_regression_input(self, img_pair):
        """
        Generates the regression input for the given image pair using the current weights.

        :param img_pair: Input image pair. Dimension: (img_width, img_height, 2)
        :return: The regression input which can be passed into the regressor.
            This is a list of inputs which may include the image pair, the bottleneck and the derived feature layers.
        """
        img_A, img_B = _img_pair_to_img_batches(img_pair)
        generator_output, *regression_input = self.generator_with_output.predict(img_A)
        # for each of the elements in regression input only select the first sample (there are only 1 samples)
        # because batches are returned
        regression_input = [batch[0] for batch in regression_input]
        if self.params['use_images']:
            regression_input.append(img_pair)

        return regression_input

    def regress_from_img(self, img_pair):
        """
        Regresses the fundamental matrix from the given image pair using the current weights.

        :param img_pair: Input image pair. Dimension: (img_width, img_height, 2)
        :return: Fundamental matrix. Dimension: (3, 3)
        """
        img_A, img_B = _img_pair_to_img_batches(img_pair)

        if self.params['use_images']:
            # as the regressor also uses images, imgA and imgB are needed
            gen_img, F = self.generator_with_regressor.predict([img_A, img_B])
        else:
            # the regressor uses no images, only input from generator and the generator only needs imgA
            gen_img, F = self.generator_with_regressor.predict(img_A)
        return F[0]  # only 1 sample in batch

    def regress_from_regression_input(self, regression_input):
        """
        Regresses the fundamental matrix from the given regression inputs using the current weights (of the regressor)

        :param regression_input: Regression input which was generated using generate_regression_input.
        :return: Fundamental matrix. Dimension: (3, 3)
        """
        regression_input = _regression_input_to_batch(regression_input)
        F_batch = self.regress_from_regression_input_batch(regression_input)
        return F_batch[0]  # only 1 sample in batch

    def regress_from_regression_input_batch(self, regression_input_batch):
        """
        Regresses the fundamental matrices for a batch of regression inputs using the current weights (of the regressor)

        :param regression_input_batch: Batch of regression inputs which were generated using generate_regression_input.
        :return: Batch of fundamental matrices. Dimension: (None, 3, 3)
        """
        F_batch = self.regressor.predict(regression_input_batch)

        return F_batch

    def train_GAN(self, img_pair, epochs, discr_iterations,
                  plot_interval=None, img_path_prefix=None, check_img_mse=False, verbose=0):
        """
        Trains the GAN for the given image pair.

        :param img_pair: Image pair to train the GAN.
        :param epochs: Number of training epochs.
        :param discr_iterations: Number of discriminator iterations in each epoch.
        :param plot_interval: How often is the generated image plot.
            - None: no plotting
            - positive integer: plot every nth epoch
            - -1: plot only after the last epoch.
        :param img_path_prefix: Prefix for plotted image files. If None: image is only plotted but not saved.
        :param check_img_mse: bool - Check and store the image mean squared error of the generated image in the history.
        :param verbose: Verbosity level: 0 to 2
        :return: History of the GAN training, dictionary of lists
        """
        return self.__do_train_gan_for_sample(img_pair, epochs=epochs, discr_iterations=discr_iterations,
                                              plot_interval=plot_interval, img_path_prefix=img_path_prefix,
                                              check_img_mse=check_img_mse, verbose=verbose)

    def train_GAN_and_regressor(self, img_pair, F_true, epochs, discr_iterations,
                                plot_interval=None, img_path_prefix=None, check_img_mse=False, verbose=0):
        """
        Train the GAN and the regressor together using a combined loss for generator and regressor.

        :param img_pair: Image pair to train.
        :param F_true: Ground truth fundamental matrix.
        :param epochs: Number of training epochs.
        :param discr_iterations: Number of discriminator iterations in each epoch.
        :param plot_interval: plot_interval: How often is the generated image plot.
            - None: no plotting
            - positive integer: plot every nth epoch
            - -1: plot only after the last epoch.
        :param img_path_prefix: Prefix for plotted image files. If None: image is only plotted but not saved.
        :param check_img_mse: bool - Check and store the image mean squared error of the generated image in the history.
        :param verbose: Verbosity level: 0 to 2
        :return: History of the training, dictionary of lists
        """
        return self.__do_train_gan_for_sample(img_pair, F_true=F_true, epochs=epochs, discr_iterations=discr_iterations,
                                              plot_interval=plot_interval, img_path_prefix=img_path_prefix,
                                              check_img_mse=check_img_mse, verbose=verbose)

    def train_regressor(self, regression_input, F_true):
        """
        Trains the regressor for the given regression input and F_true.

        The regressor is only trained for one epoch on that single sample.

        :param regression_input: Regression input to train for.
        :param F_true: Ground truth fundamental matrix.
        :return: History of the training, dictionary of lists
        """
        return self.train_regressor_batch(_regression_input_to_batch(regression_input), _F_to_F_batch(F_true))

    def train_regressor_batch(self, regression_input_batch, F_true_batch):
        """
        Trains the regressor for the given regression input and F_true batch.

        The regressor is only trained for one epoch on that batch.

        :param regression_input_batch: Batch of regression inputs to train for.
        :param F_true_batch: Batch of Ground truth fundamental matrices.
        :return: History of the training, dictionary of lists
        """
        loss = self.regressor.train_on_batch(regression_input_batch, F_true_batch)
        return loss

    def update_regressor_lr(self, update_fn):
        """
        Updates the current learning rate of the regressor using the given update function.
        The update function gets the old lr as input and should return the new lr.
        This new lr is then set as the regressor lr.

        :param update_fn: Function applied to compute new lr: update_fn(old_lr: float) -> new_lr: float
        :return: New learning rate which was set.
        """
        old_lr = float(K.get_value(self.regressor.optimizer.lr))
        new_lr = update_fn(old_lr)
        K.set_value(self.regressor.optimizer.lr, new_lr)
        return new_lr

    def save_weights(self, file_prefix=None):
        """
        Save all model weights.

        Multiple files will be stored, as this model has multiple sub models.
        All files are stored within the model folder.

        :param file_prefix: If defined, use this as prefix for the weight file names.
            If None: store temporary weights.
        """
        if file_prefix is None:
            file_prefix = TMP_WEIGHTS_FILE_PREFIX
        for i, model in enumerate(self.__models_with_weights):
            model.save_weights(self.model_folder + '/' + file_prefix + ('_%d.h5' % i))

    def load_weights(self, file_prefix=None, remove=False):
        """
        Loads all model weights.

        All files are loaded from within the model folder.

        :param file_prefix: file_prefix: If defined, use this as prefix for the weight file names.
            If None: load temporary weights.
        :param remove: If True, remove the loaded weight files.
        """
        if file_prefix is None:
            file_prefix = TMP_WEIGHTS_FILE_PREFIX
        for i, model in enumerate(self.__models_with_weights):
            file = self.model_folder + '/' + file_prefix + ('_%d.h5' % i)
            model.load_weights(file)
            if remove:
                os.remove(file)

    def plot_models(self, file_prefix):
        """
        Plots all sub models
        :param file_prefix:
        """
        print('Plotting model with file prefix %s' % file_prefix)
        plot_model(self.generator, to_file=file_prefix+'generator.png', show_shapes=True)
        plot_model(self.generator_with_output, to_file=file_prefix+'generator_with_output.png', show_shapes=True)
        plot_model(self.generator_with_regressor, to_file=file_prefix+'generator_with_regressor.png', show_shapes=True)
        plot_model(self.discriminator, to_file=file_prefix+'_discriminator.png', show_shapes=True)
        plot_model(self.regressor, to_file=file_prefix+'regressor.png', show_shapes=True)
        plot_model(self.gan, to_file=file_prefix+'gan.png', show_shapes=True)
        plot_model(self.gan_with_regressor, to_file=file_prefix+'gan_with_regressor.png', show_shapes=True)

    # verbose=0 -> no logging
    # verbose=1 -> only show current epoch
    # verbose=2 -> show epoch results and details
    # plot_interval: None -> disabled, >0 every i epochs, -1 only at the end of all epochs
    def __do_train_gan_for_sample(self, img_pair_sample, F_true=None, epochs=1, discr_iterations=1,
                                  plot_interval=None, img_path_prefix=None, check_img_mse=False,
                                  verbose=0):
        img_A, img_B = _img_pair_to_img_batches(img_pair_sample)
        if F_true is not None:
            F_true = _F_to_F_batch(F_true)

        if plot_interval is not None and img_path_prefix is not None:
            # Save original images for later debugging
            _save_imgs(img_A, img_B, img_path_prefix + 'img_A.png', img_path_prefix + 'img_B.png')

        valid = np.array([1])
        fake = np.array([0])

        generator_loss_history = []
        generator_gen_loss_history = []
        generator_F_history = []
        discriminator_history = []
        discriminator_real_history = []
        discriminator_fake_history = []
        img_mse_history = []

        fake_B = self.generator.predict(img_A)  # generate fake B for 1st epoch

        for epoch in range(1, epochs+1):
            if verbose == 1:
                print(('--> GAN epoch %d/%d' % (epoch, epochs)).ljust(100), end='\r')
            elif verbose > 1:
                print('--> GAN epoch %d/%d' % (epoch, epochs))

            # --- train discriminator
            if verbose >= 2:
                print('-----> Train D...'.ljust(100), end='\r')
            discr_input = np.concatenate([img_B, fake_B])
            discr_target = np.concatenate([valid, fake])
            for it in range(1, discr_iterations + 1):
                discriminator_loss_real, discriminator_loss_fake = self.discriminator.train_on_batch(discr_input, discr_target)
                discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2
                if verbose >= 2:
                    print('-----> D iteration %d/%d [loss: %f, real_loss: %f, fake_loss: %f]'.ljust(100) %
                          (it, discr_iterations, discriminator_loss, discriminator_loss_real, discriminator_loss_fake), end='\r')
            discriminator_history.append(discriminator_loss)
            discriminator_real_history.append(discriminator_loss_real)
            discriminator_fake_history.append(discriminator_loss_fake)

            # --- train generator
            if verbose >= 2:
                print('-----> Train G...'.ljust(100), end='\r')
            if F_true is None:
                generator_loss = self.gan.train_on_batch(img_A, valid)
                generator_loss_history.append(generator_loss)
                if verbose == 1:
                    print(('--> GAN epoch %d/%d [D - loss: %f] [G - loss: %f]' %
                          (epoch, epochs, discriminator_loss, generator_loss)).ljust(100), end='\r')
                elif verbose > 1:
                    print(('---> [D - loss: %f] [G - loss: %f]' % (discriminator_loss, generator_loss)).ljust(100))
            else:
                if self.params['use_images']:
                    loss, generator_loss, fmatrix_loss = self.gan_with_regressor.train_on_batch([img_A, img_B], [valid, F_true])
                else:
                    loss, generator_loss, fmatrix_loss = self.gan_with_regressor.train_on_batch(img_A, [valid, F_true])
                generator_loss_history.append(loss)
                generator_gen_loss_history.append(generator_loss)
                generator_F_history.append(fmatrix_loss)
                if verbose == 1:
                    print(('--> GAN epoch %d/%d [D - loss: %f] [G - loss: %f, gen_loss: %f, F_loss: %f]' %
                          (epoch, epochs, discriminator_loss, loss, generator_loss, fmatrix_loss)).ljust(100), end='\r')
                elif verbose > 1:
                    print(('--->  [D - loss: %f] [G - loss: %f, gen_loss: %f, F_loss: %f]' %
                          (discriminator_loss, loss, generator_loss, fmatrix_loss)).ljust(100))

            # Generate for next epoch and for results checking (so that the img has not to be generated twice)
            fake_B = self.generator.predict(img_A)

            if check_img_mse:
                img_mse = _calc_image_mse(img_A, img_B, fake_B)
                img_mse_history.append(img_mse)
                if verbose >= 2:
                    print('---> [image_mse: %f]' % img_mse)

            if plot_interval is not None and plot_interval != -1 and epoch % plot_interval == 0:
                if img_path_prefix is not None:
                    img_path = img_path_prefix + ('generated_B_%04d.png' % epoch)
                else:
                    img_path = None
                _plot_img(img_A, img_B, fake_B, img_path)

        if plot_interval == -1:
            if img_path_prefix is not None:
                img_path = img_path_prefix + 'generated_B.png'
            else:
                img_path = None
            _plot_img(img_A, img_B, fake_B, img_path)

        if F_true is None:
            return {'discriminator_loss': discriminator_history,
                    'discriminator_loss_real': discriminator_real_history,
                    'discriminator_loss_fake': discriminator_fake_history,
                    'generator_loss': generator_loss_history,
                    'img_mse': img_mse_history}
        else:
            return {'discriminator_loss': discriminator_history,
                    'discriminator_loss_real': discriminator_real_history,
                    'discriminator_loss_fake': discriminator_fake_history,
                    'generator_loss': generator_loss_history,
                    'generator_loss_gen': generator_gen_loss_history,
                    'generator_F_loss': generator_F_history,
                    'img_mse': img_mse_history}


def stack_regression_inputs(regression_inputs_list):
    """
    Stack regression inputs.

    Use this to make batches of regression inputs given a list of regression inputs.

    :param regression_inputs_list: List of regression inputs.
    :return: Batch of regression inputs.
    """
    assert len(regression_inputs_list) > 0

    return [np.stack(single_input_batch, axis=0) for single_input_batch in list(zip(*regression_inputs_list))]


def _calc_image_mse(img_A, img_B, gen_B):
    img_A = img_A[0, :, :, 0]
    img_B = img_B[0, :, :, 0]
    gen_B = gen_B[0, :, :, 0]

    return image_mse(img_B, gen_B)


def _save_imgs(img_A, img_B, img_A_path, img_B_path):
    img_A = img_A[0, :, :, 0]
    img_B = img_B[0, :, :, 0]

    plt.imsave(img_A_path, img_A, cmap='gray')
    plt.imsave(img_B_path, img_B, cmap='gray')


def _plot_img(img_A, img_B, gen_B, img_path):
    img_A = img_A[0, :, :, 0]
    img_B = img_B[0, :, :, 0]
    gen_B = gen_B[0, :, :, 0]

    if img_path is None:
        plt.subplot(1, 3, 1)
        plt.title('IMG A')
        plt.imshow(img_A, 'gray')

        plt.subplot(1, 3, 2)
        plt.title('IMG B - real')
        plt.imshow(img_B, 'gray')

        plt.subplot(1, 3, 3)
        plt.title('IMG B - generated')
        plt.imshow(gen_B, 'gray')

        plt.show()
    else:
        plt.imsave(img_path, gen_B, cmap='gray')


def _img_pair_to_img_batches(img_pair_sample):
    img_A, img_B = img_pair_sample[:, :, 0], img_pair_sample[:, :, 1]  # extract images from sample
    img_A, img_B = _img_to_img_batch(img_A), _img_to_img_batch(img_B)
    return img_A, img_B


def _img_to_img_batch(img):
    img = np.expand_dims(img, axis=2) # reshape to (width, height, 1)
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img


def _img_pair_to_img_batch(img_pair_sample):
    img_pair_sample = np.expand_dims(img_pair_sample, axis=0)  # add batch dimension
    return img_pair_sample


def _F_to_F_batch(F):
    F = np.expand_dims(F, axis=0)
    return F


def _regression_input_to_batch(regression_input):
    regression_input = [np.expand_dims(input_element, axis=0) for input_element in regression_input]
    return regression_input
