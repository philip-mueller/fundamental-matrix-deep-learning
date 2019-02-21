from dataset_utils.dataset_loading import load_FMatrix_dataset
from train.model.model_trainers import load_FMatrix_model


def test_model(model_folder, epoch, dataset_name, norm, metrics=[], limit=None, plot_interval=None, type='TEST', verbose=0):
    """
    Tests the model from the given model_folder with the given dataset and the given metrics.

    The model and its weights from the given epoch are loaded and then tested against the dataset.

    :param model_folder: Relative path of the model folder.
    :param epoch: Epoch for which the model weights should be loaded.
        - positive int: epoch number.
            Note that this only works when the corresponding model weights have been stored in this epoch.
        - -1: load the final weights.
            Note that this only works when the final model weights of the model have been stored.
    :param dataset_name: Name of the dataset to load.
        For this dataset the TEST dataset file is loaded with the given norm.
    :param norm: Specifies the FMatrix norm that was used in the model to test.
        This influences which dataset file is loaded.
    :param metrics: List of epipolar metric functions which the model is tested against.
    :param limit: Limit the number of test samples to be loaded. For None all samples are loaded.
    :param plot_interval: Plot interval of the generated images.
    :param verbose: Verbosity level.
    :return: Dictionary containing two entries for each given metric, one for SIFT points and ine for random sampled.
        The values of the entries represent the results of the metric averaged over the whole loaded test set.
    """
    print('Loading test data...')
    test_data_loader, file_name = load_FMatrix_dataset(dataset_name, norm, type=type, random=True, limit=limit)
    X_test, Y_test = test_data_loader.load_all()

    print('Loading model...')
    model = load_FMatrix_model(model_folder, epoch=epoch)

    print('Testing...')
    results = model.test_model(X_test, Y_test, metrics=metrics, plot_interval=plot_interval, verbose=verbose)

    print('Done')
    return results

