from train.utils.metrics import combine_metrics_history, FMatrixMetrics, FMatrixMetricsAccumulator
from train.model.model import FMatrixGanModel, stack_regression_inputs
from train.utils.params import Params
import numpy as np


from train.utils.results import load_model_results_file
from train.utils.regression_input_storage import RegressionInputStorageWriter, RegressionInputStorageReader


class FMatrixGanTrainer:
    """
    Contains the high level training methods for the model (combined and spearated)
    as well as other utils related to model training.
    """
    def __init__(self, params, model_folder, metrics=[], img_size=(128, 128)):
        """
        Inits the trainer and a model which is contained in the trainer.

        :param params: Hyperparameters
        :param model_folder: Folder path, in which all results and temporary data of the model is stored.
        :param metrics: List of epipolar metric functions which are used during training and validation.
            The results of these metrics will be included in the training history.
            As all given metrics are epipolar metrics they must be functions with the following signature:
            metric(F, points) -> float
            Where F is a single fundamental matrix and points is a list of pairs of corresponding points.
        :param img_size: (image_width, image_height), defining the size of the input images.
        """
        if not isinstance(params, Params):
            params = Params(params)
        self.params = params
        self.model_folder = model_folder

        self.model = FMatrixGanModel(params, model_folder, img_size)

        self.train_metrics = FMatrixMetrics(metrics, use_SIFT=False)
        self.train_metrics_sift = FMatrixMetrics(metrics, use_SIFT=True)
        self.val_metrics = FMatrixMetrics(metrics, prefix='val', use_SIFT=False)
        self.val_metrics_sift = FMatrixMetrics(metrics, prefix='val', use_SIFT=True)

    def fit(self, train, val, **kwargs):
        """
        Fits the model using the given training and validation data.
        The hyperparameters are set in the constructor of this class, this includes the used training method,
        e.g. combined or separated as these are defined by the hyperparams.

        :param train: Training data of type FMatrixData.
        :param val: Validation data of type FMatrixData or None if validation should not be done.
        :param kwargs: Additional arguments, depending on the used training method (separated / combined)
        :return: Training history
            Dictionary with the keys "history" and "history_details".
            - history: List of dictionaries. Each dict represents one epoch and contains the metric results
                       of the validation and training data.
            - history_details: depends on the used training method, may also be None.
        """
        if self.params['train_method'] == 'combined':
            return self.fit_combined(train, val, **kwargs)
        elif self.params['train_method'] == 'separated':
            return self.fit_separated(train, val, **kwargs)
        else:
            raise ValueError('Unkown train method: ' + self.params['train_method'])

    def fit_combined(self, train, val,
                     validate_interval=None, sample_plot_interval=None, gan_iteration_plot_interval=-1, save_imgs=False,
                     save_weights_interval=None, check_img_mse=False, detailed_results_interval=None,
                     verbose=0, **kwargs):
        """
        Fits the model using the given training and validation data by training the GAN and the regressor in combination.

        Each epoch iterates over all samples. For each sample the GAN and the regressor are trained in combination.
        For this, during GAN training the generator will be trained with a combined loss
        of the generator GAN loss and the regression loss.

        :param train: Training data of type FMatrixData.
        :param val: Validation data of type FMatrixData or None if validation should not be done.
        :param validate_interval: How often is validation done.
            None or positive int:
            - None: no validation
            - positive int: Each nth epoch validation will be done.
                            So for validate_interval=1 validation is done every epoch.
        :param sample_plot_interval: How often are the generated images plotted or saved.
            Whether they are saved or plotted depends on save_imgs.
            None or int:
            - None: Images are not plotted or saved
            - positive int: The generated images are plotted every nth sample in every epoch.
                            Which generated image is plot depends on gan_iteration_plot_interval
            - -1: Images are plotted/saved only for the last sample in each epoch.
        :param gan_iteration_plot_interval: The generated images of which GAN epoch of a sample are plotted/saved.
            This is only relevant if sample_plot_interval enabled plotting/saving (is not None)
            Whether they are saved or plotted depends on save_imgs.
            None or int:
            - None: Images are not plotted or saved
            - positive int: The generated image of every nth GAN training epoch of a saved/plotted sample is saved/plotted
            - -1: Only the image generated after the GAN has been trained for a sample is saved.
        :param save_imgs: Specifies whether images should be plotted in the current python session or saved as files.
            If true: the image is saved, if false it is plotted.
            Note which images are saved or plotted depends on sample_plot_interval and gan_iteration_plot_interval.
        :param save_weights_interval: How often should the model weights be stored on disk.
            None or int:
            - None: Do not store the model weights
            - positive int: Store the model weights every nth epoch.
            - -1: Store weights at the end of the training (after the last epoch).
        :param check_img_mse: Specifies whether the generated images should be compared to the true images using
            mean-squared error. If true, this will be done for every GAN epoch for every sample in every epoch
            and the results are stored in the history_details.
        :param detailed_results_interval: How often should the detailed history of the GAN training be recorded.
            None or positive int:
            - None: Do not record detailed histories, only the normal history is recorded
            - positive int: Record the detailed history for every nth sample in every epoch.
        :param verbose: Verbosity level
            verbose=0 -> only show current epoch
            verbose=1 -> show current sample in epoch and validation data, not details to gan training
            verbose=2 -> show data for each sample
            verbose=3 -> show all details
        :return: Training history
            Dictionary with the keys "history" and "history_details".
            - history: List of dictionaries. Each dict represents one epoch and contains the metric results
                       of the validation and training data.
            - history_details: dict with only key "combined_epochs" which contain a list of entries for all epochs.
                               Each entry contains the GAN training histories of all samples for which it was recorded.
                               Which GAN trainings are recorded depends on detailed_results_interval.
                               The value of "combined_epochs" is None if no detailed histories were recorded.
        """
        if len(kwargs) > 0:
            print('Ignoring the following arguments passed to fit method: %s' % kwargs)

        self.params['train_method'] = 'combined'  # Set train method so that the results can be correctly analyzed

        fit_epochs = self.params['fit_epochs']
        metric_results_list = []
        detailed_epoch_histories = []
        for epoch in range(1, fit_epochs + 1):
            if verbose == 0:
                print(('Combined Epoch %d/%d' % (epoch, fit_epochs)).ljust(100), end='\r')
            elif verbose > 0:
                print(('Combined Epoch %d/%d' % (epoch, fit_epochs)).ljust(100))

            # train
            metrics_result, detailed_sample_histories = \
                self.__fit_combined_epoch(train, val, epoch=epoch,
                                          sample_plot_interval=sample_plot_interval,
                                          gan_iteration_plot_interval=gan_iteration_plot_interval,
                                          validate_interval=validate_interval,
                                          save_imgs=save_imgs, save_weights_interval=save_weights_interval,
                                          check_img_mse=check_img_mse,
                                          detailed_results_interval=detailed_results_interval,
                                          verbose=verbose)
            metric_results_list.append(metrics_result)
            if detailed_results_interval is not None:
                detailed_epoch_histories.append(detailed_sample_histories)

            if verbose >= 1:
                print(str(metrics_result).ljust(100))

        # Save model weights
        if save_weights_interval is not None and save_weights_interval == -1:
            self.model.save_weights('weights_final')

        metrics_history = combine_metrics_history(metric_results_list)
        history_details = dict(combined_epochs=detailed_epoch_histories)
        return dict(history=metrics_history, history_details=history_details)

    def __fit_combined_epoch(self, train, val, epoch,
                             sample_plot_interval, validate_interval, gan_iteration_plot_interval, save_imgs,
                             save_weights_interval, check_img_mse, detailed_results_interval, verbose):
        num_samples = len(train)

        # Prepare epoch
        train.randomize()
        epoch_F_loss = 0.0
        metrics_accumulator = FMatrixMetricsAccumulator(self.train_metrics)
        metrics_accumulator_sift = FMatrixMetricsAccumulator(self.train_metrics_sift)

        # Train epoch
        detailed_sample_histories = []
        for i, (img_pair, F_true) in enumerate(iter(train)):
            sample_nr = i+1
            if verbose == 1:
                print(('> Sample %d/%d' % (sample_nr, num_samples)).ljust(100), end='\r')
            elif verbose > 1:
                print(('> Sample %d/%d' % (sample_nr, num_samples)).ljust(100))

            # Check whether images should be plotted/saved for this sample
            img_name = '/epoch_%04d_sample_%04d_' % (epoch, sample_nr)
            img_path_prefix, gan_plot_interval = _get_plot_interval_and_img_prefix(self.model_folder, img_name,
                                                                                   sample_nr, num_samples,
                                                                                   sample_plot_interval,
                                                                                   gan_iteration_plot_interval,
                                                                                   save_imgs)

            # Train GAN and regressor combined
            gan_train_history = self.model.train_GAN_and_regressor(img_pair, F_true,
                                                                   epochs=self.params['sample_train_epochs'],
                                                                   discr_iterations=self.params['train_discr_iterations'],
                                                                   plot_interval=gan_plot_interval,
                                                                   img_path_prefix=img_path_prefix,
                                                                   check_img_mse=check_img_mse,
                                                                   verbose=verbose-1)

            # Train only the regressor a few more iterations
            if self.params['regressor_train_iterations'] is not None and self.params['regressor_train_iterations'] > 0:
                regression_history = self.__train_regressor(img_pair, F_true,
                                                            num_iterations=self.params['regressor_train_iterations'],
                                                            verbose=verbose)
            else:
                regression_history = {}

            if detailed_results_interval is not None and sample_nr % detailed_results_interval == 0:
                detailed_sample_histories.append(
                    dict(combined_gan=gan_train_history, regressor_iterations=regression_history))

            F_History = gan_train_history['generator_F_loss']
            epoch_F_loss += F_History[-1]  # only last sample matters

            # regress FMatrix for using in the metrics
            F_pred = self.model.regress_from_img(img_pair)
            metrics_accumulator.add_sample(img_pair, F_true, F_pred)
            metrics_accumulator_sift.add_sample(img_pair, F_true, F_pred)

        # Save model weights
        if save_weights_interval is not None and save_weights_interval > 0 and epoch % save_weights_interval == 0:
            self.model.save_weights('weights_epoch_%04d' % epoch)

        # Train Metrics
        epoch_F_loss /= num_samples
        train_metric_results = metrics_accumulator.calculate_for_accumulated_samples()
        train_metrics_sift_results = metrics_accumulator_sift.calculate_for_accumulated_samples()

        # Validation Metrics
        val_metric_sift_results, val_metric_results = self.__validate_epoch(self.__validate_with_img_pairs,
                                                                            val=val,
                                                                            epoch=epoch,
                                                                            validate_interval=validate_interval,
                                                                            verbose=verbose)

        metrics_result = dict(F_loss=epoch_F_loss,
                              **train_metric_results, **train_metrics_sift_results,
                              **val_metric_results, **val_metric_sift_results)
        return metrics_result, detailed_sample_histories

    def __train_regressor(self, img_pair, F_true, num_iterations, verbose):
        if verbose >= 2:
            print('-> Train regressor'.ljust(100))
        regression_input = self.model.generate_regression_input(img_pair)

        regression_loss_history = []
        for i in range(num_iterations):
            iteration = i+1
            loss = self.model.train_regressor(regression_input, F_true)
            regression_loss_history.append(loss)

            if verbose == 2:
                print(('--> Iteration %d/%d [F_loss: %f]' % (iteration, num_iterations, loss)).ljust(100), end='\r')
            elif verbose >= 3:
                print(('--> Iteration %d/%d [F_loss: %f]' % (iteration, num_iterations, loss)).ljust(100))
        return dict(F_loss=regression_loss_history)

    def fit_separated(self, train, val,
                      validate_interval=None, sample_plot_interval=None, gan_iteration_plot_interval=-1,
                      save_imgs=False, save_weights_interval=None, check_img_mse=False, keep_reg_input_values=False,
                      detailed_results_interval=None, reg_input_model_folder=None, verbose=0, **kwargs):
        """
        Fits the model using the given training and validation data by training the GAN and the regressor separately.

        Before the actual training, the GAN is trained for all samples.
        For each sample first the GAN is trained and then the needed input for the regressor
        is generated using the generator (bottleneck and derived features if used).
        All those regressor input are stored. Then the stored values will be split into batches.
        With these batches the regressor will be trained several epochs without training the GAN again.
        :param train: Training data of type FMatrixData.
        :param val: Validation data of type FMatrixData or None if validation should not be done.
        :param validate_interval: How often is validation done.
            None or positive int:
            - None: no validation
            - positive int: Each nth epoch validation will be done.
                            So for validate_interval=1 validation is done every epoch.
        :param sample_plot_interval: How often are the generated images plotted or saved.
            Whether they are saved or plotted depends on save_imgs.
            None or int:
            - None: Images are not plotted or saved
            - positive int: The generated images are plotted every nth sample in every epoch.
                            Which generated image is plot depends on gan_iteration_plot_interval
            - -1: Images are plotted/saved only for the last sample in each epoch.
        :param gan_iteration_plot_interval: The generated images of which GAN epoch of a sample are plotted/saved.
            This is only relevant if sample_plot_interval enabled plotting/saving (is not None)
            Whether they are saved or plotted depends on save_imgs.
            None or int:
            - None: Images are not plotted or saved
            - positive int: The generated image of every nth GAN training epoch of a saved/plotted sample is saved/plotted
            - -1: Only the image generated after the GAN has been trained for a sample is saved.
        :param save_imgs: Specifies whether images should be plotted in the current python session or saved as files.
            If true: the image is saved, if false it is plotted.
            Note which images are saved or plotted depends on sample_plot_interval and gan_iteration_plot_interval.
        :param save_weights_interval: How often should the model weights be stored on disk.
            None or int:
            - None: Do not store the model weights
            - positive int: Store the model weights every nth epoch.
            - -1: Store weights at the end of the training (after the last epoch).
        :param check_img_mse: Specifies whether the generated images should be compared to the true images using
            mean-squared error. If true, this will be done for every GAN epoch for every sample in every epoch
            and the results are stored in the history_details.
        :param keep_reg_input_values: Specifies whether the intermediate regression inputs values which have
            been written to files should be kept or removed after the training is finished.
            If True: keeps the files. If False (default) removes them.
        :param detailed_results_interval: How often should the detailed history of the GAN training be recorded.
            None or positive int:
            - None: Do not record detailed histories, only the normal history is recorded
            - positive int: Record the detailed history for every nth sample in every epoch.
        :param reg_input_model_folder: Specifies another model folder from which the regression inputs should be used.
            The model folder path is given relative to the current folder.
            If this parameter is set, then no regression inputs are created but the inputs from the other model
            are directly loaded and then the regressor is trained.
            If this is None (default), then the regression inputs are created normally.
            Important note: This parameter should only be used during hyperparameter tuning when only the
            regressor parameters are tuned. The weights from the GAN will stay untrained when using this.
            So this should nt be used for the final training of the model!
        :param verbose: Verbosity level
            verbose=0 -> only show current epoch
            verbose=1 -> show current sample in epoch and validation data, not details to gan training
            verbose=2 -> show data for each sample
            verbose=3 -> show all details
        :return: Training history
            Dictionary with the keys "history" and "history_details".
            - history: List of dictionaries. Each dict represents one epoch and contains the metric results
                       of the validation and training data.
            - history_details: dict with only key "gan_samples" which contain the histories of all
                               the GAN trainings of all samples for which they where recorded.
                               For which samples the GAN trainings are recorded depends on detailed_results_interval.
                               The value of "gan_samples" is [] if no detailed histories were recorded.
        """
        if len(kwargs) > 0:
            print('Ignoring the following arguments passed to fit method: %s' % kwargs)

        self.params['train_method'] = 'separated'  # Set train method so that the results can be correctly analyzed

        # --- Params ---
        regressor_epochs = self.params['regressor_epochs']
        batch_size = self.params['regressor_batch_size']

        if reg_input_model_folder is None:
            # --- Train generator and generate regression inputs ---
            print('Preparing regression inputs for training data...'.ljust(100))

            reg_input_reader, detailed_sample_histories =\
                self.__train_generator_for_regression_inputs(train,
                                                             sample_plot_interval=sample_plot_interval,
                                                             gan_iteration_plot_interval=gan_iteration_plot_interval,
                                                             save_imgs=save_imgs, check_img_mse=check_img_mse,
                                                             detailed_results_interval=detailed_results_interval,
                                                             verbose=verbose)
            # --- Prepare for validation: Train generator and generate regression inputs ---
            # regression inputs will be stored in files
            val_reg_input_reader = self.__prepare_regression_inputs_for_validation(val, validate_interval, verbose)
        else:
            print('WARNING: Skipping creation of regression inputs and GAN training. '
                  'Using Regression inputs from model folder ' + reg_input_model_folder)
            reg_input_reader = RegressionInputStorageReader(reg_input_model_folder, 'TRAIN')
            val_reg_input_reader = RegressionInputStorageReader(reg_input_model_folder, 'VAL')
            detailed_sample_histories = None

        # --- Train regressor ---
        metric_results_list = []

        for epoch in range(1, regressor_epochs + 1):
            if verbose == 0:
                print(('Regressor Epoch %d/%d' % (epoch, regressor_epochs)).ljust(100), end='\r')
            elif verbose > 0:
                print(('Regressor Epoch %d/%d' % (epoch, regressor_epochs)).ljust(100))

            # Train regressor epoch
            metrics_result = self.__fit_regressor_epoch(train=train, reg_input_reader=reg_input_reader,
                                                        val=val, val_reg_input_reader=val_reg_input_reader,
                                                        epoch=epoch, batch_size=batch_size,
                                                        validate_interval=validate_interval,
                                                        save_weights_interval=save_weights_interval,
                                                        verbose=verbose)
            metric_results_list.append(metrics_result)

            if verbose >= 1:
                print(str(metrics_result).ljust(100))

        # Save model weights
        if save_weights_interval is not None and save_weights_interval == -1:
            self.model.save_weights('weights_final')

        if not keep_reg_input_values:
            reg_input_reader.remove_files()
            if val_reg_input_reader is not None:
                val_reg_input_reader.remove_files()

        metrics_history = combine_metrics_history(metric_results_list)
        history_details = dict(gan_samples=detailed_sample_histories)
        return dict(history=metrics_history, history_details=history_details)

    def __train_generator_for_regression_inputs(self, train,
                                                sample_plot_interval, gan_iteration_plot_interval,
                                                save_imgs, check_img_mse, detailed_results_interval, verbose,
                                                reg_input_file_prefix='TRAIN'):
        
        train.randomize()

        # regression inputs are written into files so that python does not run out of memory
        reg_input_writer = RegressionInputStorageWriter(self.model_folder, reg_input_file_prefix,
                                                        random_indices=train.random_indices)
        num_samples = len(train)

        detailed_sample_histories = []
        for i, (img_pair, F_true) in enumerate(iter(train)):
            sample_nr = i + 1
            if verbose == 1:
                print(('> Sample %d/%d' % (sample_nr, num_samples)).ljust(100), end='\r')
            elif verbose > 1:
                print(('> Sample %d/%d' % (sample_nr, num_samples)).ljust(100))

            # Check whether images should be plotted/saved for this sample
            img_name = 'regressor_sample_%04d_' % sample_nr
            img_path_prefix, gan_plot_interval = _get_plot_interval_and_img_prefix(self.model_folder, img_name,
                                                                                   sample_nr, num_samples,
                                                                                   sample_plot_interval,
                                                                                   gan_iteration_plot_interval,
                                                                                   save_imgs)

            # Train GAN
            history = self.model.train_GAN(img_pair, epochs=self.params['sample_train_epochs'],
                                           discr_iterations=self.params['train_discr_iterations'],
                                           plot_interval=gan_plot_interval,
                                           img_path_prefix=img_path_prefix,
                                           check_img_mse=check_img_mse,
                                           verbose=verbose - 1)
            if detailed_results_interval is not None and sample_nr % detailed_results_interval == 0:
                detailed_sample_histories.append(history)

            # Generate regression input
            regression_input = self.model.generate_regression_input(img_pair)
            reg_input_writer.store_regression_input(regression_input)

        return reg_input_writer.create_reader(), detailed_sample_histories

    def __fit_regressor_epoch(self, train, reg_input_reader, val, val_reg_input_reader,
                              epoch, batch_size, validate_interval, save_weights_interval, verbose):
        train.randomize()
        reg_input_reader.randomize(train.random_indices)  # reg inputs need to have the same order as train data

        train_batches = train.batches(batch_size)
        reg_input_batches = reg_input_reader.batches(batch_size)

        assert len(train_batches) == len(reg_input_batches)
        num_batches = len(train_batches)

        epoch_F_loss = 0.0
        metrics_accumulator = FMatrixMetricsAccumulator(self.train_metrics)
        metrics_accumulator_sift = FMatrixMetricsAccumulator(self.train_metrics_sift)

        # iterate over batches
        for i, (regression_input_batch, train_batch) in enumerate(zip(reg_input_batches, train_batches)):
            batch_nr = i + 1
            img_pair_batch, F_batch = train_batch

            if verbose == 1:
                print(('> Batch %d/%d' % (batch_nr, num_batches)).ljust(100), end='\r')
            elif verbose > 1:
                print(('> Batch %d/%d' % (batch_nr, num_batches)).ljust(100))

            # Train
            batch_F_loss = self.model.train_regressor_batch(regression_input_batch, F_batch)
            epoch_F_loss += batch_F_loss

            # Regress F matrices for metrics
            F_pred_batch = self.model.regress_from_regression_input_batch(regression_input_batch)

            if verbose == 2:
                print(('-> [F_loss: %f]' % batch_F_loss).ljust(100), end='\r')
            elif verbose > 2:
                print(('-> [F_loss: %f]' % batch_F_loss).ljust(100))

            # Metrics
            metrics_accumulator.add_batch(img_pair_batch, F_batch, F_pred_batch)
            metrics_accumulator_sift.add_batch(img_pair_batch, F_batch, F_pred_batch)

        # Update regressor learning rate after each epoch when lr_R_decay is used
        regressor_lr_decay = self.params['lr_R_decay']
        if regressor_lr_decay is not None:
            new_lr = self.model.update_regressor_lr(lambda old_lr: regressor_lr_decay * old_lr)
            if verbose >= 1:
                print(('Note: updated regressor lr to value %f' % new_lr).ljust(100))

        # Save model weights
        if save_weights_interval is not None and save_weights_interval > 0 and epoch % save_weights_interval == 0:
            self.model.save_weights('weights_epoch_%04d' % epoch)

        # Train Metrics
        epoch_F_loss /= num_batches
        train_metric_results = metrics_accumulator.calculate_for_accumulated_samples()
        train_metrics_sift_results = metrics_accumulator_sift.calculate_for_accumulated_samples()

        # Validation Metrics
        val_metric_sift_results, val_metric_results = self.__validate_epoch(self.__validate_with_regression_inputs,
                                                                            val=val,
                                                                            val_reg_input_reader=val_reg_input_reader,
                                                                            epoch=epoch,
                                                                            validate_interval=validate_interval,
                                                                            verbose=verbose)
        metric_result = dict(F_loss=epoch_F_loss,
                             **train_metric_results, **train_metrics_sift_results,
                             **val_metric_results, **val_metric_sift_results)

        return metric_result

    def __validate_epoch(self, val_fn, epoch, validate_interval=1, verbose=0, **args):
        if validate_interval is not None and all([val is not None for val in args.values()]):  # all validation data is set
            if epoch % validate_interval == 0:
                if verbose >= 1:
                    print('> Validate'.ljust(100))
                val_metric_sift_results, val_metric_results = val_fn(verbose=verbose, **args)
            else:
                # add results but for all metrics the value is None
                val_metric_results = self.val_metrics.get_empty_metrics_result()
                val_metric_sift_results = self.val_metrics_sift.get_empty_metrics_result()
        else:
            # add empty results (no validation metrics)
            val_metric_results = {}
            val_metric_sift_results = {}
        return val_metric_sift_results, val_metric_results

    def __validate_with_img_pairs(self, val, verbose=0):
        val_metrics_accumulator = FMatrixMetricsAccumulator(self.val_metrics)
        val_metrics_accumulator_sift = FMatrixMetricsAccumulator(self.val_metrics_sift)

        num_samples = len(val)
        for (i, val_sample) in enumerate(val):
            img_pair, F_true = val_sample
            F_pred = self.predict_fmatrix(img_pair, verbose=verbose)

            val_metrics_accumulator.add_sample(img_pair, F_true, F_pred)
            val_metrics_accumulator_sift.add_sample(img_pair, F_true, F_pred)

        val_metric_results = val_metrics_accumulator.calculate_for_accumulated_samples()
        val_metric_sift_results = val_metrics_accumulator_sift.calculate_for_accumulated_samples()
        return val_metric_sift_results, val_metric_results

    def __prepare_regression_inputs_for_validation(self, val, validate_interval, verbose):
        if validate_interval is not None:
            print('Preparing regression inputs for validation data...'.ljust(100))
            self.model.save_weights()
            val_reg_input_reader, detail_history = \
                self.__train_generator_for_regression_inputs(val,
                                                             sample_plot_interval=None,
                                                             gan_iteration_plot_interval=None,
                                                             check_img_mse=False, save_imgs=False, verbose=verbose,
                                                             detailed_results_interval=None,
                                                             reg_input_file_prefix='VAL')
            self.model.load_weights(remove=True)
        else:
            val_reg_input_reader = None
        return val_reg_input_reader

    def __validate_with_regression_inputs(self, val, val_reg_input_reader, verbose=0):
        batch_size = self.params['regressor_batch_size']
        reg_input_val_batches = val_reg_input_reader.batches(batch_size)
        val_batches = val.batches(batch_size)

        val_metrics_accumulator = FMatrixMetricsAccumulator(self.val_metrics)
        val_metrics_accumulator_sift = FMatrixMetricsAccumulator(self.val_metrics_sift)

        for i, (val_batch, reg_input_batch) in enumerate(zip(val_batches, reg_input_val_batches)):
            img_batch, F_true_batch = val_batch
            F_pred_batch = self.model.regress_from_regression_input_batch(reg_input_batch)
            val_metrics_accumulator.add_batch(img_batch, F_true_batch, F_pred_batch)
            val_metrics_accumulator_sift.add_batch(img_batch, F_true_batch, F_pred_batch)

        val_metric_results = val_metrics_accumulator.calculate_for_accumulated_samples()
        val_metric_sift_results = val_metrics_accumulator_sift.calculate_for_accumulated_samples()
        return val_metric_sift_results, val_metric_results

    def predict_fmatrix_batch(self, img_pair_batch, plot_interval=None, verbose=0):
        """
        Predicts the fundamental matrices for the given image pair batch.

        Note that the weights are all restored after each prediction, so that the predictions do not influence
        each other or future predictions.

        :param img_pair_batch: Image pair batch for which to predict the matrices.
            np.array of dimension (None, width, height, 2)
        :param plot_interval:
        :param verbose:
        :return: Predicted fundamental matrices
            np.array of dimension (None, 3, 3)
        """
        num_samples = len(img_pair_batch)
        predicted_Fs = []
        for i, img_pair in enumerate(img_pair_batch):
            if verbose > 0:
                print(('-> Predicting Sample %d/%d' % (i+1, num_samples)).ljust(100))
            F_pred, _ = self.predict_fmatrix(img_pair, plot_interval, verbose=verbose)
            predicted_Fs.append(F_pred)
        return np.array(predicted_Fs)

    def predict_fmatrix(self, img_pair, plot_interval=None, verbose=0):
        """
        Predicts a single fundamental matrix from the given image pair.

        Note that the weights are all restored after the prediction, so that it does not influence other prediction.

        :param img_pair: Image pair for which to predict the matrix.
            np.array of dimension (width, height, 2)
        :param plot_interval:
        :param verbose:
        :return: (F, history)
            - F: the predicted FMatrix, np.array of dimension (3, 3)
            - history: List of dicts representing the training history of the GAN which was trained for prediction.
        """
        self.model.save_weights()
        history = self.model.train_GAN(img_pair, epochs=self.params['sample_predict_epochs'],
                                       discr_iterations=self.params['predict_discr_iterations'],
                                       plot_interval=plot_interval,
                                       verbose=verbose-1)
        F = self.model.regress_from_img(img_pair)
        self.model.load_weights(remove=True)
        return F, history

    def test_model(self, X_test, Y_test, metrics, plot_interval=None, verbose=0):
        """
        Tests the model against the given test data using the given metrics.
        :param X_test: Test image pairs. np.array of dimension (None, width, height, 2)
        :param Y_test: Test ground truth matrices. np.array of dimension (None, 3, 3).
        :param metrics: List of epipolar metrics to use.
        :param plot_interval:
        :param verbose:
        :return: Dictionary where each key represents a metric. Note that each metric
            is tested with SIFT and with random sampled point correspondences.
        """
        test_metrics = FMatrixMetrics(metrics, use_SIFT=False, prefix='test')
        test_metrics_sift = FMatrixMetrics(metrics, use_SIFT=True, prefix='test')

        Y_pred = self.predict_fmatrix_batch(X_test, plot_interval=plot_interval, verbose=verbose)

        results = test_metrics.calculate_for_batch(X_test, Y_test, Y_pred)
        sift_results = test_metrics_sift.calculate_for_batch(X_test, Y_test, Y_pred)
        return dict(**results, **sift_results)

    def load_model_weights(self, epoch=-1):
        """
        Loads the model weights for this model.
        Note that only the weights are loaded, other states like the of the optimizer are not loaded.

        :param epoch: Epoch for which to load the weight. -1 to load the final weights.
        """
        if epoch == -1:
            self.model.load_weights('weights_final')
        else:
            self.model.load_weights('weights_epoch_%04d' % epoch)


def _get_plot_interval_and_img_prefix(model_folder, img_name, sample_nr, num_samples,
                                      sample_plot_interval, gan_iteration_plot_interval, save_imgs):
    if sample_plot_interval is not None and \
            ((sample_plot_interval != -1 and sample_nr % sample_plot_interval == 0) or  # plot every sample_plot_interval samples
             (sample_plot_interval == -1 and sample_nr == num_samples)):  # only plot the last sample
        gan_plot_interval = gan_iteration_plot_interval
        if save_imgs:
            img_path_prefix = model_folder + '/' + img_name
        else:
            img_path_prefix = None
    else:
        gan_plot_interval = None
        img_path_prefix = None
    return img_path_prefix, gan_plot_interval


def load_FMatrix_model(model_folder, epoch=-1, metrics=[]):
    """
    Loads a trained FMatrix model.
    Note that only the weights and hyperparams are loaded, other states like the of the optimizer are not loaded.

    :param model_folder: Folder from which to load the model. Relative to current folder.
    :param epoch: Epoch for which to load the weight. -1 to load the final weights.
    :param metrics: List of metrics to use in the model. Only relevant when the model should be trained further.
    :return: The loaded model (class FMatrixGanTrainer).
    """
    p = load_model_results_file(model_folder)['params']
    model = FMatrixGanTrainer(p, model_folder, metrics)
    model.load_model_weights(epoch)
    return model
