import math
from abc import ABCMeta, abstractmethod

import numpy as np
import os
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.preprocessing import image


datasets_path = 'datasets'


def get_dataset_folder(dataset_name):
    """
    Returns the dataset folder for the given dataset.
    :param dataset_name: Name of the dataset.
    :return: Dataset folder path (relative).
    """
    return datasets_path + '/' + dataset_name


def get_dataset_file(dataset_name, filename):
    """
    Returns the file path of a file in a dataset.
    :param dataset_name: Name of the dataset.
    :param filename: Name of the file.
    :return: Dataset file path (relative).
    """
    return get_dataset_folder(dataset_name) + '/' + filename


def find_dataset_file(dataset_name, file_type):
    """
    Finds the dataset file of the given type.

    If no corresponding file was found, AssertionsError is raised.
    :param dataset_name: Name of the dataset.
    :param file_type: File type: either 'par', 'ang' or 'fmatrix' if it exists
    :return: File name of the dataset file if it was found.
    """
    file_names = [f for f in os.listdir(get_dataset_folder(dataset_name))
                  if f.endswith('_' + file_type + '.txt') or f.endswith('_' + file_type + '.csv')]
    if len(file_names) == 1:
        return file_names[0]
    elif len(file_names) == 0:
        raise AssertionError('No camera parameter file found for dataset "%s" (type: %s).' % (dataset_name, file_type))
    else:
        raise AssertionError(
            'Multiple camera parameter files found for dataset "%s".' %
            dataset_name)


def get_normed_FMatrix_dataset_type(norm):
    """
    Returns the name of the dataset file type for FMatrix data with the given norm.
    :param norm: Name of the norm, 'fro' or 'abs'. Or None for no norm.
    :return: Name of the dataset file type. E.g. to use it in find_dataset_file.
    """
    if norm is None:
        return 'fmatrix'
    else:
        return 'fmatrix_' + norm


def _read_and_map_dataset(dataset_name, file_name, mapping_fn, delmiter):
    """
    Utility function to read a dataset and map it to a dataset.

    :param dataset_name: Name of the dataset.
    :param file_name: Name of the dataset file.
    :param mapping_fn: Function to map a row to the resulting dataset row.
    :param delmiter: Delimiter used to separate values in the file.
    :return: List of mapped dataset rows.
    """
    file = get_dataset_folder(dataset_name) + '/' + file_name
    raw_data = np.genfromtxt(file, delimiter=delmiter, dtype=None, skip_header=1, encoding='utf8')
    dataset = []
    for row in raw_data:
        dataset.append(mapping_fn(list(row)))
    return dataset


def read_camera_parameter_dataset_file(dataset_name, file_name, delmiter=' '):
    """
    Reads the dataset file containing the image names with their camera params.

    -------------- Structure of a row in the file (, as delimiter): -----------------
    image_file,
    k11, k12, k13, k21, k22, k23, k31, k32, k33,
    r11, r12, r13, r21, r22, r23, r31, r32, r33,
    t1, t2, t3
    ---------------------------------------------------------------------------------

    :param dataset_name: Name of the dataset.
    :param file_name: Name of the dataset file.
    :param delmiter: Delimiter used to separate values in the file.
    :return: List of dicts where each dict represents a sample. Each dict has the keys: img, K, R, t.
    """
    def camera_mapping_fn(row):
        K = np.array(row[1:10]).reshape((3, 3))
        R = np.array(row[10:19]).reshape((3, 3))
        t = np.array(row[19:22]).reshape((3))
        return {'img': row[0], 'K': K, 'R': R, 't': t}

    return _read_and_map_dataset(dataset_name, file_name, camera_mapping_fn, delmiter=delmiter)


def write_camera_parameter_dataset_file(dataset, dataset_name, file_name, delmiter=' '):
    """
    Writes the given camera parameter dataset (image names and camera params) into a dataset file.

    -------------- Structure of a row in the file (, as delimiter): -----------------
    image_file,
    k11, k12, k13, k21, k22, k23, k31, k32, k33,
    r11, r12, r13, r21, r22, r23, r31, r32, r33,
    t1, t2, t3
    ---------------------------------------------------------------------------------

    :param dataset: Dataset rows as a list of dicts, each row has the keys: img, K, R, t.
    :param dataset_name: Name of the dataset. Will be newly created.
    :param file_name: Name of the dataset file to be created.
    :param delmiter: Delimiter used to separate values in the file.
    """
    file = get_dataset_folder(dataset_name) + '/' + file_name
    if os.path.exists(file):
        print('camer parameter file "%s" already exists. Skipping creation' % file)
        return

    file_data = []
    for row in dataset:
        file_data.append([row['img']] +
                         row['K'].flatten().tolist() +
                         row['R'].flatten().tolist() +
                         row['t'].flatten().tolist())
    header = 'img,K11,K12,K13,K21,K22,K23,K31,K32,K33,R11,R12,R13,R21,R22,R23,R31,R32,R33,t1,t2,t3'
    np.savetxt(file, file_data, delimiter=delmiter, header=header, encoding='utf8', fmt='%s')


def read_fmatrix_dataset_file(dataset_name, file_name, delmiter=','):
    """
    Reads the fmatrix dataset file containing two image names and the corresponding fundamental matrix in each row.

    -------------- Structure of a row in the file (, as delimiter): -----------------
    image_file_A, image_file_B
    f11, f12, f13, f21, f22, f23, f31, f32, f33
    ---------------------------------------------------------------------------------

    :param dataset_name: Name of the dataset.
    :param file_name: Name of the dataset file.
    :param delmiter: Delimiter used to separate values in the file.
    :return: List of dicts where each dict represents a sample. Each dict has the keys: img_A, img_B, F.
    """
    def fmatrix_mapping_fn(row):
        F = np.array(row[2:11]).reshape((3, 3))
        return {'img_A': row[0], 'img_B': row[1], 'F': F}

    return _read_and_map_dataset(dataset_name, file_name, fmatrix_mapping_fn, delmiter=delmiter)


def write_fmatrix_dataset_file(dataset, dataset_name, file_name, delmiter=','):
    """
    Writes the given fmatrix dataset (two image names and fundamentl matrix in each row) into a dataset file.

    -------------- Structure of a row in the file (, as delimiter): -----------------
    image_file_A, image_file_B
    f11, f12, f13, f21, f22, f23, f31, f32, f33
    ---------------------------------------------------------------------------------

    :param dataset: Dataset rows as a list of dicts, each row has the keys: img_A, img_B, F.
    :param dataset_name: Name of the dataset. Will be newly created.
    :param file_name: Name of the dataset file to be created.
    :param delmiter: Delimiter used to separate values in the file.
    """
    file = get_dataset_folder(dataset_name) + '/' + file_name
    if os.path.exists(file):
        print('fmatrix file "%s" already exists. Skipping creation' % file)
        return

    file_data = []
    for row in dataset:
        file_data.append([row['img_A'], row['img_B']] + row['F'].flatten().tolist())
    header = 'img_A,img_B,F11,F12,F13,F21,F22,F23,F31,F32,F33'
    np.savetxt(file, file_data, delimiter=delmiter, header=header, encoding='utf8', fmt='%s')


def load_image(dataset_name, img_file_name, target_size=(128, 128), normalize=True):
    """
    Loads an image from the given dataset and normalizes it to [0, 1].

    :param dataset_name: Name of the dataset, that contains the image.
    :param img_file_name: Filename of the image within the dataset.
    :param target_size: Target image size. Note: If this is not the original image size, interpolation
        will be done, which may lead to values other than 0 or 1 in binary images.
    :param normalize: True to normalize the image (divide each pixel by 255)
    :return: Loaded image as np.array. Each pixel has values in [0, 1].
    """
    img_path = get_dataset_folder(dataset_name) + '/' + img_file_name
    img = image.load_img(img_path, grayscale=True, target_size=target_size, interpolation='nearest')
    img_array = image.img_to_array(img, data_format='channels_last')
    if normalize:
        img_array /= 255  # normalize pixel values to either be 0 or 1 instead of 0 or 255.
    return img_array.reshape(target_size)


class DataLoader(Sequence):
    """
    Abstract base class for data loaders, which load a dataset.

    This class contains utils for randomized indexing, batch creation and limiting of dataset length.
    """
    ___metaclass__ = ABCMeta

    @abstractmethod
    def _get_element(self, data_index): pass

    def _batch_fn(self, batch):
        """
        Function applied to create batches of data.
        :param batch: List of samples used in this batch.
        :return: Batch created from the list of samples.
        """
        return np.array(batch)

    def __init__(self, num_samples, random=True):
        """
        Init
        :param num_samples: Number of samples that the laoded dataset has.
        :param random: True, to randomize data loading.
        """
        self.num_samples = num_samples
        self.random = random
        self.random_indices = None
        if random:
            self.randomize()
        else:
            self.restore_original_order()

    def randomize(self, random_indices=None):
        """
        Randomizes the dataset.
        :param random_indices: If None, a new random order is created.
            If a list of indices is given, this list will be used as the random order.
        """
        if random_indices is None:
            self.random_indices = np.random.permutation(len(self))
        else:
            assert len(random_indices) == len(self)
            self.random_indices = random_indices

    def restore_original_order(self):
        """
        Restores the original dataset order, meaning that no randomization is used anymore.
        """
        self.random_indices = list(range(len(self)))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get_element(self.random_indices[item])
        elif isinstance(item, slice):
            return [self[i] for i in range(*item.indices(self.num_samples)) if i < self.num_samples]
        else:
            raise TypeError

    def __iter__(self):
        return CountIterator(self)

    def batches(self, batch_size):
        """
        Creates a batch data loader, which creates batches on top of this dataset.
        :param batch_size: Batch size of each batch.
        :return: BatchDataLoader
        """
        return BatchDataLoader(self, batch_size, random=self.random, batch_fn=self._batch_fn)

    def on_epoch_end(self):
        if self.random:
            self.randomize()

    def limit(self, num_limit_samples):
        """
        Limits the number of loaded samples.
        A maximum of num_limit_samples is loaded.

        :param num_limit_samples: Maximum number of loaded samples.
        """
        self.num_samples = min(self.num_samples, num_limit_samples)

    def load_all(self):
        """
        Loads the whole dataset (or the limited dataset, if limit was set).

        :return: The whole dataset as a single batch. The output was processed by _batch_fn.
        """
        return self._batch_fn(list(self))


class CountIterator:
    """
    Iterator which can be used with any container class (e.g. a list) and iterates over that container
    using a counting index.
    """
    def __init__(self, data):
        """
        Init.
        :param data: Container object which contains the data to be iterated over.
        """
        self.data = data
        self.length = len(data)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.length:
            element = self.data[self.index]
            self.index += 1
            return element
        else:
            raise StopIteration


class BatchDataLoader(Sequence):
    """
    Class for loading batches from a data loader.

    Batches are created on the fly so data can be loaded lazily.
    """
    def __init__(self, data_loader, batch_size, random=True, batch_fn=lambda x: x):
        """
        Inits the batch data loader.
        :param data_loader: Data loader to load the samples. Must be a subclass of DataLoader.
        :param batch_size: Use batch size.
        :param random: True to randomize the batches and samples in the batches.
        :param batch_fn: Function applied to a list of samples in a batch which returns the created batch.
        """
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.batch_fn = batch_fn
        self.random = random

    def __len__(self):
        return int(np.ceil(len(self.data_loader) / float(self.batch_size)))

    def __getitem__(self, item):
        """
        Returns the batch(es) for the given index or range.
        :param item: Index or range of the batch.
        :return: Created batch.
        """
        if isinstance(item, int):
            batch_data = self.data_loader[(item * self.batch_size):((item + 1) * self.batch_size)]
            batch_data = self.batch_fn(batch_data)
            return batch_data
        else:
            raise TypeError('BatchDataLoader does not support slice indexing')

    def __iter__(self):
        return CountIterator(self)

    def on_epoch_end(self):
        if self.random:
            self.data_loader.randomize()


class FMatrixData(DataLoader):
    """
    Dataloader for loading a fundamental matrix dataset.
    """
    def __init__(self, dataset_name, file_name, random=True, normalize_imgs=True):
        """
        Inits the dataloader.

        The dataset file is directly loaded into memory. The images will be loaded lazily.

        :param dataset_name: Name of the dataset to load.
        :param file_name: Name of the dataset file in that dataset to load.
        :param random: True to randomize the sample order.
        :param normalize_imgs: True to normalize the image (divide each pixel by 255).
        """
        self.dataset_name = dataset_name
        self.normalize_imgs = normalize_imgs
        self.__data = np.array(read_fmatrix_dataset_file(dataset_name, file_name))
        num_samples = len(self.__data)
        super().__init__(num_samples, random=random)

    def _get_element(self, data_index, load_x=True):
        row = self.__data[data_index]
        y = np.array(row['F'])
        if load_x:
            x = np.stack([load_image(self.dataset_name, row['img_A'], normalize=self.normalize_imgs),
                          load_image(self.dataset_name, row['img_B'], normalize=self.normalize_imgs)], axis=2)
            return x, y
        else:
            return y

    def _batch_fn(self, batch):
        # Convert list of (x, y) to (X, Y)
        X, Y = tuple(zip(*batch))
        return np.array(X), np.array(Y)

    def label_only_loader(self):
        """
        Returns a data loader which only loads the labels for this fundamental matrix dataset but not the images.
        :return: FMatrixLabelDataLoader
        """
        return FMatrixLabelDataLoader(self)


class FMatrixLabelDataLoader(DataLoader):
    """
    Dataloader for loading the labels (fundamental matrixes) of a fundamental matrix dataset, but not the images.
    """
    def __init__(self, fmatrix_data_loader):
        """
        Init.
        :param fmatrix_data_loader: FMatrixData of which the labels should be laoded.
        """
        self.fmatrix_data_loader = fmatrix_data_loader
        super().__init__(len(fmatrix_data_loader), random=fmatrix_data_loader.random)
        # Same random order as fmatrix_data_loader
        assert fmatrix_data_loader.random_indices is not None
        self.randomize(random_indices=fmatrix_data_loader.random_indices)

    def _get_element(self, data_index):
        return self.fmatrix_data_loader.__get_element(data_index, load_x=False)


def load_FMatrix_data(params, limit_samples=None, limit_val_samples=None, use_train_as_val=False, validate=True,
                      random=True, normalize_imgs=True, **kwargs):
    """
    Loads the FMatrix dataset for training.

    This loads the TRAIN and the VAL dataset as both are needed during training.

    :param params: Dict containing the keys 'dataset' for the dataset name and 'norm' for the used norm.
    :param limit_samples: If this is not None, limits the number of loaded TRAIN samples.
    :param limit_val_samples: If this is not None, limits the number of loaded VAL samples.
    :param use_train_as_val: If True, uses the training dataset also for validation.
    :param validate: If this is True, load a validation dataset, if False no VAL dataset is loaded (None is returned)
    :param random: True to use random order in the datasets.
    :param normalize_imgs: True to normalize the images (divide each pixel by 255).
    :param kwargs:
    :return: (train, val) the TRAIN and the VAL dataset. Both are FMatrixData objects.
    """
    dataset_name = params['dataset']
    norm = params['norm']

    train_postfix = 'TRAIN'
    train, train_file_name = load_FMatrix_dataset(dataset_name, norm, train_postfix, random=random, limit=limit_samples,
                                                  normalize_imgs=normalize_imgs)
    print('Loaded %d TRAIN samples from %s' % (len(train), train_file_name))
    if validate:
        if use_train_as_val:
            val_postfix = 'TRAIN'
        else:
            val_postfix = 'VAL'
        val, val_file_name = load_FMatrix_dataset(dataset_name, norm, val_postfix,
                                                  random=random, limit=limit_val_samples,
                                                  normalize_imgs=normalize_imgs)
        print('Loaded %d VAL samples from %s' % (len(val), val_file_name))
    else:
        val = None

    return train, val


def load_FMatrix_dataset(dataset_name, norm=None, type='TRAIN', random=True, limit=None, normalize_imgs=True):
    """
    Loads a FMatrix dataset.

    The dataset file is automatically determined for the given norm and type.

    :param dataset_name: Name of the dataset to load.
    :param norm: Norm to use, either None, 'fro' or 'abs'.
    :param type: Type of dataset file to loade. Either 'TRAIN', 'VAL' or 'TEST'.
    :param random: True to use random order.
    :param limit: If not None, limits the number of loaded samples.
    :param normalize_imgs: True to normalize the images (divide each pixel by 255).
    :return: (data, file_name) where data is a FMatrixData object for the laoded dataset and file_name
        the name of the loaded file.
    """
    file_name = find_dataset_file(dataset_name, get_normed_FMatrix_dataset_type(norm) + '_' + type)
    data = FMatrixData(dataset_name, file_name, random=random, normalize_imgs=normalize_imgs)
    if limit is not None:
        data.limit(limit)
    return data, file_name
