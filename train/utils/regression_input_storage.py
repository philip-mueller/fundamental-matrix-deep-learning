import gc
import os

import numpy as np

from dataset_utils.dataset_loading import DataLoader
from train.model.model import stack_regression_inputs

REGRESSION_INPUT_SAVE_FOLDER = 'regression_inputs'


class RegressionInputStorageWriter:
    """
    Utility class to store the intermediate regression inputs on disk.

    Each regression input will be stored in its own file.
    """
    def __init__(self, model_folder, prefix, random_indices=None):
        """
        Inits the writer.

        :param model_folder: model folder to store values in.
        :param prefix: Prefix for regression input files.
        :param random_indices: Random indices used in the data loader, so that regression inputs can be stored
            in the same order as the X and Y data although randomization is used.
            None, if no randomization is used.
        """
        self.model_folder = model_folder
        self.prefix = prefix
        self.__current_index = 0
        if not os.path.exists(self.model_folder + '/' + REGRESSION_INPUT_SAVE_FOLDER):
            os.makedirs(self.model_folder + '/' + REGRESSION_INPUT_SAVE_FOLDER)
        _remove_files(self.model_folder, self.prefix)  # delete any previous files

        self.random_indices = random_indices

    def __get_file_index(self):
        """
        Returns the index of the next regression input that will be written.
        :return: Next index. Includes the random_indices to map to the correct real index.
        """
        if self.random_indices is None:
            return self.__current_index
        else:
            return self.random_indices[self.__current_index]

    def store_regression_input(self, regression_input):
        """
        Stores the given regression input to disk in its own file.

        The index is increased automatically.
        :param regression_input: Regression input, list of np.arrays.
        """
        stored_index = self.__get_file_index()
        file = self.model_folder + '/' + REGRESSION_INPUT_SAVE_FOLDER + \
               ('/%s_reg_input_%08d.npz' % (self.prefix, stored_index))

        np.savez(file, *regression_input)
        del regression_input
        gc.collect()
        self.__current_index += 1

    def create_reader(self, random=True):
        """
        Creates a reader for this writer.

        This reader can then read the written regression inputs.
        :param random: True if the reader should use randomization.
        :return: The created reader.
        """
        reader = RegressionInputStorageReader(self.model_folder, self.prefix, random=random)
        reader.randomize(self.random_indices)
        return reader


class RegressionInputStorageReader(DataLoader):
    """
    Utility class to read the intermediate regression inputs from disk.
    """
    def __init__(self, model_folder, prefix, random=True):
        """
        Inits the writer.

        :param model_folder: model folder where values are stored in.
        :param prefix: Prefix for regression input files.
        :param random: True if values should be read in random order.
        """
        self.model_folder = model_folder
        self.prefix = prefix
        num_samples = len(self.__get_all_filenames())
        super().__init__(num_samples, random=random)

    def __get_all_filenames(self):
        """
        Returns all relevant regression input files.
        :return: All relevant regression input files.
        """
        return _get_filenames(self.model_folder, self.prefix)

    def _get_element(self, data_index):
        """
        Reads the regression input file with the given data_index.
        :param data_index: The real index (not the randomized) of the regression input file.
        :return: Regression input. List of np.arrays.
        """
        file = self.model_folder + '/' + REGRESSION_INPUT_SAVE_FOLDER + \
               ('/%s_reg_input_%08d.npz' % (self.prefix, data_index))
        with np.load(file) as data:
            sorted_files = ['arr_{}'.format(i) for i in range(len(data.files))]
            regression_input = [data[key] for key in sorted_files]
            return regression_input

    def _batch_fn(self, batch):
        """
        Stacks the regression inputs to make a batch.
        :param batch: List of values in the batch.
        :return: Created batch.
        """
        return stack_regression_inputs(batch)

    def remove_files(self):
        """
        Removes the stored regression inputs from disk.
        """
        _remove_files(self.model_folder, self.prefix)


def _get_filenames(model_folder, prefix):
    return [file for file in os.listdir(model_folder + '/' + REGRESSION_INPUT_SAVE_FOLDER)
            if file.startswith(prefix)]


def _remove_files(model_folder, prefix):
    for file_name in iter(_get_filenames(model_folder, prefix)):
        os.remove(model_folder + '/' + REGRESSION_INPUT_SAVE_FOLDER + '/' + file_name)
