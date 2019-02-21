import numpy as np
import os
import urllib.request
import shutil
import zipfile

from dataset_utils import fmatrix
from dataset_utils.dataset_loading import read_camera_parameter_dataset_file, read_fmatrix_dataset_file, \
    write_fmatrix_dataset_file, get_dataset_folder, \
    datasets_path, find_dataset_file, get_dataset_file, get_normed_FMatrix_dataset_type
from dataset_utils.fmatrix import all_norms
from dataset_utils.generation_camera_params import camera_dataset_names, camera_configs, mesh_files
from dataset_utils.img_processing import create_silhouette_dataset_from_img_dataset, resize_images
from dataset_utils.dataset_generation import generate_synthetic_dataset


def download_camera_dataset(dataset_name):
    """
    Downloads the camera dataset with the given name from 'http://vision.middlebury.edu/mview/data/data/'
    and saves it in the datasets folder.

    :param dataset_name: Name of the dataset.
    """
    print('\n============================== Dataset download ==============================')
    if os.path.exists(get_dataset_folder(dataset_name)):
        print('Dataset "%s" already exists. Skipping download.' % dataset_name)
        return

    base_url = 'http://vision.middlebury.edu/mview/data/data/'
    url = base_url + dataset_name + '.zip'
    zip_file_name = datasets_path + '/' + dataset_name + '.zip'

    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    print('Downloading dataset from "%s" into file "%s"...' % (url, zip_file_name))
    with urllib.request.urlopen(url) as response, open(zip_file_name, 'wb') as zip_file:
        shutil.copyfileobj(response, zip_file)
    print('Done.')

    print('Unpacking zip file...')
    zip_file = zipfile.ZipFile(zip_file_name, 'r')
    zip_file.extractall('datasets')
    zip_file.close()
    os.remove(zip_file_name)
    check_dataset(dataset_name)
    print('Done.')


def check_dataset(dataset_name):
    """
    Checks that the given dataset folder exists and conatins a .par file.
    :param dataset_name: Name of the dataset.
    """
    dataset_path = get_dataset_folder(dataset_name)
    if not os.path.exists(dataset_path):
        raise AssertionError('Expected dataset folder for dataset "%s" does not exist.' % dataset_name)

    find_dataset_file(dataset_name, 'par')


def camera_dataset_to_fmatrix_dataset(camera_dataset):
    """
    Converts the given camera dataset into a fundamental matrix dataset.

    The fundamental matrix dataset is created by combining the camera dataset with itsself.
    Rows with 0-matrices are ignored.

    :param camera_dataset: Loaded camera dataset. List of dicts with keys img, K, R, t.
        Each list item represents an image with filename "img" with its camera params "K", "R", "t".
    :return: Fundamental matrix dataset. List of dicts with "img_A", "img_B" as the two image file names and
        "F" containing an np.array of the fundamental matrix.
    """
    fmatrix_dataset = []
    for A in camera_dataset:
        for B in camera_dataset:
            # ignore combination of same dataset
            if A is B:
                continue

            F = fmatrix.fundamental_matrix_from_camera(A['K'], A['R'], A['t'], B['K'], B['R'], B['t'])
            if np.all(F == 0):
                # both images have been made form the same perspective => ignore
                continue

            fmatrix_dataset.append({
                'img_A': A['img'],
                'img_B': B['img'],
                'F': F
            })
    return fmatrix_dataset


def normalize_fmatrix_dataset(fmatrix_dataset, norm):
    """
    Normalizes the given dataset using the given norm.
    :param fmatrix_dataset: Fundamental matrix dataset. List of dicts with "img_A", "img_B" as the two image file names and
        "F" containing an np.array of the fundamental matrix.
    :param norm: Name of the norm: 'fro' or 'abs', or None to not normalize.
    :return: Normalized dataset. Same as fmatrix_dataset except that matrices are normed.
    """
    if norm is None:
        return fmatrix_dataset

    def norm_row(row):
        return {
            'img_A': row['img_A'],
            'img_B': row['img_B'],
            'F': fmatrix.normalize(row['F'], norm)
        }

    return list(map(norm_row, fmatrix_dataset))


def create_fmatrix_files_from_camera_parameter_file(dataset_name, camera_parameter_file_name=None,
                                                    fmatrix_file_names=None,
                                                    norms=None):
    """
    Creates a fundamental matrix dataset from an existing camera parameter dataset.

    The created dataset is included in the same dataset folder and has the same name. Only
    an additional parameter file is created.

    :param dataset_name: Name of the dataset for which the fundamental matrix file should be created.
    :param camera_parameter_file_name: Name of the camera params file.
    :param fmatrix_file_names: List of filenames of the fmatrix files for the norms.
        If None default names are created.
    :param norms: List of norms ('fro', 'abs', None) for which the files should be created. One file for each norm.
    """
    print('\n============================== Fundamental matrix file creation ==============================')
    if camera_parameter_file_name is None:
        camera_parameter_file_name = find_dataset_file(dataset_name, 'par')

    if norms is None:
        norms = fmatrix.all_norms

    if fmatrix_file_names is None:
        fmatrix_file_names = [dataset_name + '_' + get_normed_FMatrix_dataset_type(norm) + '.csv' for norm in norms]

    if not len(fmatrix_file_names) == len(norms):
        raise ValueError('fmatrix_file_names and norms have to be of same length')

    # check if files already exist
    if all(map(lambda file_name: os.path.exists(get_dataset_folder(dataset_name) + '/' + file_name),
               fmatrix_file_names)):
        print('All fmatrix files "%s" already exist. Skipping creation' % fmatrix_file_names)
        return

    print('Reading camera dataset file "%s" in dataset "%s".' % (camera_parameter_file_name, dataset_name))
    camera_dataset = read_camera_parameter_dataset_file(dataset_name, camera_parameter_file_name)

    print('Combining images and calculating fundamental matrices...')
    fmatrix_dataset = camera_dataset_to_fmatrix_dataset(camera_dataset)
    print('Done.')

    for i, norm in enumerate(norms):
        print('Normalizing fmatrix dataset for norm: %s' % norm)
        normed_dataset = normalize_fmatrix_dataset(fmatrix_dataset, norm)
        write_fmatrix_dataset_file(normed_dataset, dataset_name, fmatrix_file_names[i])
        print('FMatrix dataset file "%s" created in dataset "%s".' % (fmatrix_file_names[i], dataset_name))


def create_silhouette_dataset_from_camera_parameter_image_dataset(source_dataset_name, source_camera_file_name=None,
                                                                  silhouette_dataset_name=None, fmatrix_file_name=None):
    """
    Creates a silhouette image fundamental matrix dataset from a texture image camera params dataset.

    :param source_dataset_name: Name of the source dataset (needs to exist in datasets folder).
    :param source_camera_file_name: Name of the source camera params file.
    :param silhouette_dataset_name: Name of the created dataset. None: default name is created.
    :param fmatrix_file_name: Name of the created fmatrix dataset file.
    :return: Used silhouette_dataset_name.
    """
    if silhouette_dataset_name is None:
        silhouette_dataset_name = source_dataset_name + '_silhouette'

    create_silhouette_dataset_from_img_dataset(source_dataset_name, silhouette_dataset_name)

    create_fmatrix_files_from_camera_parameter_file(silhouette_dataset_name,
                                                    source_camera_file_name,
                                                    fmatrix_file_name)

    return silhouette_dataset_name


def combine_fmatrix_datasets(dataset_names, new_dataset_name):
    """
    Combines (concatenated) multiple fundamental matrix datasets into one large dataset.
    The fmatrix files are concatenated and the images are copied.

    :param dataset_names: List of dataset names. All need to be fmatrix datasets in the datasets folder.
    :param new_dataset_name: Name of the created dataset.
    """
    print('\n============================== Combining datasets==============================')
    print('The following datasets are combined: \n%s' % dataset_names)
    # create new dataset folder
    new_dataset_folder = get_dataset_folder(new_dataset_name)
    if os.path.exists(new_dataset_folder):
        print('Dataset "%s" already exists. Skipping creation.' % new_dataset_name)
        return
    os.makedirs(new_dataset_folder)

    # copy images
    print('Copying images...')
    image_dict = {}
    img_counter = 0
    for dataset_name in dataset_names:
        dataset_folder = get_dataset_folder(dataset_name)
        img_files = [f for f in os.listdir(dataset_folder) if f.endswith('.png')]
        for img_file in img_files:
            new_file = 'img_%04d.png' % img_counter
            shutil.copyfile(dataset_folder + '/' + img_file, new_dataset_folder + '/' + new_file)
            image_dict[(dataset_name, img_file)] = new_file
            img_counter += 1

    # combine datasets
    for norm in all_norms:
        print('Combining datasets for norm %s...' % norm)
        all_datasets = []
        for dataset_name in dataset_names:
            file_name = find_dataset_file(dataset_name, get_normed_FMatrix_dataset_type(norm))
            dataset = read_fmatrix_dataset_file(dataset_name, file_name)
            for sample in iter(dataset):
                if (dataset_name, sample['img_A']) not in image_dict or (dataset_name, sample['img_B']) not in image_dict:
                    print('Either the image "%s" or the image "%s" was not found => sample will be ignored.' %
                          (sample['img_A'], sample['img_B']))
                    continue

                sample['img_A'] = image_dict[(dataset_name, sample['img_A'])]
                sample['img_B'] = image_dict[(dataset_name, sample['img_B'])]
                all_datasets.append(sample)

        print('Combined %d samples' % len(all_datasets))
        new_dataset_file = new_dataset_name + '_' + get_normed_FMatrix_dataset_type(norm) + '.csv'
        print('Writing to file "%s"' % new_dataset_file)
        write_fmatrix_dataset_file(all_datasets, new_dataset_name, new_dataset_file)
        print('Done.')


def split_fmatrix_dataset_all_norms(dataset_name):
    """
    Splits the dataset files for all norms into TRAIN, VAL and TEST.

    :param dataset_name: Name of the dataset to split.
    """
    for norm in all_norms:
        split_fmatrix_dataset(dataset_name, norm=norm)


def split_fmatrix_dataset(dataset_name, dataset_file_name=None, train_file_name=None, val_file_name=None,
                          test_file_name=None,
                          train_ratio=0.8, val_ratio=0.1, seed=230, norm=None):
    """
    Splits the fundamental matrix dataset for the given norm into TRAIN, VAL and TEST dataset.
    The images are not copied. Only new dataset files are created in the original dataset folder.
    The dataset is randomized before splitting,

    :param dataset_name: Dataset for which the file should be split.
    :param dataset_file_name: Name of the dataset file. None: file is found automatically.
    :param train_file_name: Name of the created TRAIN file. None: Default name is created.
    :param val_file_name:  Name of the created VAL file. None: Default name is created.
    :param test_file_name:  Name of the created TEST file. None: Default name is created.
    :param train_ratio: Ratio (float <= 1) of the train set to the full set.
    :param val_ratio: Ratio (float <= 1) of the val set to the full set.
    :param seed: Random seed used for randomization of the dataset before split.
    :param norm: Norm for which the dataset is split. The dataset for the given file need to already exist.
    :return: train_file_name, val_file_name, test_file_name
    """
    print('\n==================== Dataset splitting (TRAIN, VAL, TEST) ====================')

    if train_ratio + val_ratio > 1.0:
        raise ValueError('The sum of train_ratio and val_ratio needs to be less then 1')
    test_ratio = 1.0 - train_ratio - val_ratio

    file_postfix = get_normed_FMatrix_dataset_type(norm)
    if dataset_file_name is None:
        dataset_file_name = find_dataset_file(dataset_name, file_postfix)
    if train_file_name is None:
        train_file_name = dataset_name + '_' + file_postfix + '_TRAIN.csv'
    if val_file_name is None:
        val_file_name = dataset_name + '_' + file_postfix + '_VAL.csv'
    if test_file_name is None:
        test_file_name = dataset_name + '_' + file_postfix + '_TEST.csv'

    # Check whether files already exist
    if os.path.exists(get_dataset_file(dataset_name, train_file_name)) and \
            os.path.exists(get_dataset_file(dataset_name, val_file_name)) and \
            os.path.exists(get_dataset_file(dataset_name, test_file_name)):
        print('All files already exist ("%s, %s, %s"). Skipping dataset splitting' %
              (train_file_name, val_file_name, test_file_name))
        return train_file_name, val_file_name, test_file_name

    full_dataset = read_fmatrix_dataset_file(dataset_name, dataset_file_name)
    dataset_size = len(full_dataset)

    print('Splitting dataset "%s" (file: "%s") with ratio (TRAIN: %f, VAL: %f, TEST: %f)'
          % (dataset_name, dataset_file_name, train_ratio, val_ratio, test_ratio))

    np.random.seed(seed)
    np.random.shuffle(full_dataset)

    train_val_split = int(train_ratio * dataset_size)
    val_test_split = int((train_ratio + val_ratio) * dataset_size)

    train_dataset = full_dataset[:train_val_split]
    val_dataset = full_dataset[train_val_split:val_test_split]
    test_dataset = full_dataset[val_test_split:]

    print('FULL size: %d' % dataset_size)
    print('TRAIN size: %d' % len(train_dataset))
    print('VAL size: %d' % len(val_dataset))
    print('TEST size: %d' % len(test_dataset))

    write_fmatrix_dataset_file(train_dataset, dataset_name, train_file_name)
    write_fmatrix_dataset_file(val_dataset, dataset_name, val_file_name)
    write_fmatrix_dataset_file(test_dataset, dataset_name, test_file_name)
    print('TRAIN dataset written into file "%s"' % train_file_name)
    print('VAL dataset written into file "%s"' % val_file_name)
    print('TEST dataset written into file "%s"' % test_file_name)

    return train_file_name, val_file_name, test_file_name


def split_normed_fmatrix_datasets(dataset_name, norms=None):
    """
    Splits the fmatrix datasets for the given norms into TRAIN, VAL and TEST.
    :param dataset_name: Name of the dataset to split.
    :param norms: List of norms or None for all norms.
    :return: List of tuples (train_file_name, val_file_name, test_file_name) conatining the created file names
        for the norms.
    """
    if norms is None:
        norms = fmatrix.all_norms

    return list(map(lambda norm: split_fmatrix_dataset(dataset_name, norm=norm), norms))


def _print_dataset_info(dataset_name, file_names):
    print('\n================================================================================')
    print('Prepared dataset "%s"' % dataset_name)
    for names in iter(file_names):
        print('TRAIN: %s' % names[0])
        print('VAL: %s' % names[1])
        print('TEST: %s' % names[2])

    print('================================================================================')


def prepare_fmatrix_image_dataset_from_web(dataset_name, img_size=(128, 128)):
    print('================================================================================')
    print('Preparing image dataset from web: %s' % dataset_name)
    print('================================================================================')

    download_camera_dataset(dataset_name)
    resize_images(dataset_name, img_size)
    create_fmatrix_files_from_camera_parameter_file(dataset_name)
    file_names = split_normed_fmatrix_datasets(dataset_name)

    _print_dataset_info(dataset_name, file_names)
    return dataset_name, file_names


def prepare_fmatrix_silhouette_dataset_from_web(dataset_name, img_size=(128, 128)):
    """
    Prepares a fundamental matrix dataset using a texture dataset from the web.

    :param dataset_name: Name of the dataset from the web.
    :param img_size: Image size tuple.
    :return: silhouette_dataset_name, file_names.
    """
    print('================================================================================')
    print('Preparing silhouette dataset from web: %s' % dataset_name)
    print('================================================================================')

    download_camera_dataset(dataset_name)
    silhouette_dataset_name = create_silhouette_dataset_from_camera_parameter_image_dataset(dataset_name)
    resize_images(silhouette_dataset_name, img_size)
    file_names = split_normed_fmatrix_datasets(silhouette_dataset_name)

    _print_dataset_info(silhouette_dataset_name, file_names)
    return silhouette_dataset_name, file_names


def prepare_synthetic_fmatrix_silhouette_dataset(mesh_file, camera_config_name, dataset_name=None, img_size=(128, 128)):
    """
    Prepares a synthetic fundamental matrix dataset.

    :param mesh_file: Mesh to use. There needs to be an entry for this file in the camera_configs and
        a mesh file with the given name + .ply postfix needs to exist in backup/3d_models.
    :param camera_config_name: Name of the used camera configuration. Needs to be present in
        the mesh_file entry of camera_configs.
    :param dataset_name: Name of the dataset to create. For None a default name is created.
    :param img_size: Size of the image to create (width, height).
    :return: dataset_name, file_names
    """
    if mesh_file not in camera_configs:
        raise ValueError('No camera config for mesh %s. Known meshes: %s' % (mesh_file, camera_configs.keys()))

    mesh_camera_configs = camera_configs[mesh_file]
    if camera_config_name not in mesh_camera_configs:
        raise ValueError(
            'Unknown camera config "%s". Known configs are %s' % (camera_config_name, mesh_camera_configs.keys()))

    if dataset_name is None:
        dataset_name = 'synthetic_' + mesh_file + '_' + camera_config_name

    print('================================================================================')
    print('Preparing synthetic silhouette dataset : %s using 3D mesh file "%s" and camera configuration %s'
          % (dataset_name, mesh_file, camera_config_name))
    print('================================================================================')

    camera_param_ranges = mesh_camera_configs[camera_config_name]
    generate_synthetic_dataset(dataset_name, mesh_file, camera_param_ranges, img_size)
    create_fmatrix_files_from_camera_parameter_file(dataset_name)
    file_names = split_normed_fmatrix_datasets(dataset_name)

    _print_dataset_info(dataset_name, file_names)
    return dataset_name, file_names


def prepare_all_fmatrix_image_datasets_from_web():
    """
    Prepares fundamental matrix datasets using all texture datasets defined in camera_dataset_names.
    """
    for dataset_name in iter(camera_dataset_names):
        prepare_fmatrix_image_dataset_from_web(dataset_name)


def prepare_all_fmatrix_silhouette_datasets_from_web():
    """
    Prepares fundamental matrix silhouette image datasets using all texture datasets defined in camera_dataset_names.
    """
    for dataset_name in iter(camera_dataset_names):
        prepare_fmatrix_silhouette_dataset_from_web(dataset_name)


def prepare_all_synthetic_fmatrix_silhouette_datasets(img_size=(128, 128)):
    """
    Prepares synthetic datasets for all 3d-models and camera configs defined in camera_configs.
    :param img_size: Image size used for all of the datasets (width, height).
    """
    for mesh in iter(mesh_files):
        for camera_config in iter(camera_configs):
            prepare_synthetic_fmatrix_silhouette_dataset(mesh, camera_config, img_size=img_size)
