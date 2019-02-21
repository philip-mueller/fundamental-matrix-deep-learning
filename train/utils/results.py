import csv
import shutil

import numpy as np
import os
import json
import re

from train.utils.params import Params
from train.utils.plot import plot_metrics, report_model_params, report_model_metrics, plot_metrics_to_files

MODELS_FOLDER = 'models'
RESULTS_FILE_NAME = 'model_results.json'


def save_results(model_folder, params, history_results):
    """
    Saves the results of a model into a results json file.

    :param model_folder: Folder of the model. Here the json file is saved.
    :param params: Params object or dict of the model.
    :param history_results: History results of the training. Dict with keys history and optionally history_details.
        The history entry is a dict of lists of floats where each entry represents the history of a metric.
        history_details is dependent of the training method and may be None.
    """
    if isinstance(params, Params):
        params = params.to_dict()
    results = dict(params=params, history=history_results['history'], history_details=history_results.get('history_details', None))
    with open(model_folder + '/' + RESULTS_FILE_NAME, 'w') as results_file:
        json.dump(results, results_file, cls=NumpyJsonEncoder)


def _get_model_base_folder(model_name, timestamp=None, base_folder=None):
    if base_folder is None:
        base_folder = MODELS_FOLDER
    path = base_folder + '/' + model_name
    if timestamp is not None:
        path += '/' + timestamp
    return path


def get_model_folder(model_name, timestamp, params_version, base_folder=None):
    """
    Returns the default model folder for the given model.

    :param model_name: Name of the root model.
    :param timestamp: Timestamp of the trained model.
    :param params_version: Parameter version (used during hyperparam tuning) of the specific model.
    :param base_folder: Base folder of all model. If None the default base folder is used.
    :return: Path of the model folder.
    """
    return _get_model_base_folder(model_name, timestamp, base_folder=base_folder) + '/' + str(params_version)


def load_all_model_results(model_base_folder=None):
    """
    Loads the results of all models currently stored.

    :param model_base_folder: Base folder of all models. If None the default base folder is used.
    :return: List of model results. Each item in the list is a
        dict with entries "params", "history" and "history_details".
    """
    if model_base_folder is None:
        model_base_folder = MODELS_FOLDER

    all_results = []
    for model_name in os.listdir(model_base_folder):
        timestamps = [ts for ts in os.listdir(_get_model_base_folder(model_name, base_folder=model_base_folder))]
        unflattened_model_results = [load_model_results(model_name, timestamp, base_folder=model_base_folder)
                                     for timestamp in timestamps]
        # flatten the list
        model_results = [result for timestamp_results in unflattened_model_results for result in timestamp_results]

        all_results += model_results
    return all_results


def load_last_model_results(model_name, base_folder=None):
    """
    Loads the results of all models for the most recent training of the given root model.

    :param model_name: Name of the root model.
    :param base_folder: Base folder of all models. If None the default base folder is used.
    :return: List of model results. Each item in the list is a
        dict with entries "params", "history" and "history_details".
    """
    timestamps = [ts for ts in os.listdir(_get_model_base_folder(model_name))]
    timestamps.sort(reverse=True)
    if len(timestamps) == 0:
        raise AssertionError('No models found')

    last_timestamp = timestamps[0]
    print('----------> Loading model %s for timestamp %s <----------' % (model_name, last_timestamp))
    return load_model_results(model_name, last_timestamp, base_folder=base_folder)


def load_model_results(model_name, timestamp, base_folder=None):
    """
    Loads the results of all models of a given training (hyperparam tuning).
    Only the models where a results file exists are loaded.

    :param model_name: Name of the root model.
    :param timestamp: Timestamp of the training.
    :param base_folder: Base folder of all models. If None the default base folder is used.
    :return: List of model results. Each item in the list is a
        dict with entries "params", "history" and "history_details".
    """
    folder = _get_model_base_folder(model_name, timestamp, base_folder=base_folder)

    model_results = [load_param_model_results(model_name, timestamp, param_folder, base_folder=base_folder)
                     for param_folder in os.listdir(folder)
                     if os.path.exists(folder + '/' + param_folder + '/' + RESULTS_FILE_NAME)]

    return model_results


def load_param_model_results(model_name, timestamp, params_version, base_folder=None):
    """
    Loads the results of a single model.

    :param model_name: Name of the root model.
    :param timestamp: Timestamp of the trained model.
    :param params_version: Parameter version (used during hyperparam tuning) of the specific model.
    :param base_folder: Base folder of all model. If None the default base folder is used.
    :return: Loaded results. Dict with entries "params", "history" and "history_details".
    """
    folder = get_model_folder(model_name, timestamp, params_version, base_folder=base_folder)
    return load_model_results_file(folder)


def load_model_results_file(folder):
    """
    Loads the model results json file from the given folder.

    :param folder: Folder in which the results file should be.
    :return: Loaded results. Dict with entries "params", "history" and "history_details".
    """
    results_file_path = folder + '/' + RESULTS_FILE_NAME
    with open(results_file_path, 'r') as results_file:
        return json.load(results_file)


def load_extracted_results(folder):
    """
    Loads the results of multiple models which have been extracted into a single folder.
    This folder contains multiple model_results__....json files. Each of these files is loaded.

    :param folder: Folder with the json files.
    :return: List of model results. Each item in the list is a
        dict with entries "params", "history" and "history_details" for each of the loaded files.
    """
    results_files = [file_name for file_name in os.listdir(folder) if re.match('model_results__.*\\.json', file_name)]
    results = []
    for result_file_name in results_files:
        results_file_path = folder + '/' + result_file_name
        with open(results_file_path, 'r') as results_file:
            results.append(json.load(results_file))
    return results


def get_ordered_epoch_indices(history, metric):
    """
    Returns the indices of the epochs ordered ascending by the given metric.

    :param history: History dict where each item is a list of floats.
    :param metric: Name of the metric used to order the epochs.
    :return: List of ordered indices (1 based indices).
    """
    indices = np.argsort(history[metric])
    indices += 1
    return indices


def get_best_epoch(history, best_metric):
    """
    Finds the best epoch in the given history dict ordered by the given best_metric.

    :param history: History dict where each item is a list of floats.
    :param best_metric: Name of the metric used to order the epochs.
    :return: Dict with entries "epoch" and "metrics".
        - "epoch" is the index of the best epoch (1 based indices).
        - "metrics" is a dict with an item for each metric where the values are the metric values of the best epoch.
    """
    ordered_epochs = get_ordered_epoch_indices(history, best_metric)
    best_epoch = ordered_epochs[0]

    epoch_metrics = {metric: history[metric][best_epoch-1] for metric in history.keys()}

    return {'epoch': best_epoch, 'metrics': epoch_metrics}


def get_ordered_models(loaded_model_results, best_metric):
    """
    Orders the loaded model results by the given best_metric.
    The best epochs of each of the models are compared.

    :param loaded_model_results: List of model results. Each item in the list is a
        dict with entries "params", "history" and "history_details" for each of the loaded files.
    :param best_metric: Name of the metric used to order the epochs and model results.
    :return: List of dicts with entries "params", "history" and "best_epoch". Each dict in the list represents
        the results of a model. The results are ordered starting with the best model (lowest metric values).
        - params: Dict of params of the model.
        - history: History dict where each item is a list of floats.
        - best_epoch: Dict with entries "epoch" which contains the 1-based index of the best epoch
            and "metrics" which is a dict containing the metric values of the best epoch.
    """
    params = [result['params'] for result in loaded_model_results]
    histories = [result['history'] for result in loaded_model_results]
    best_epochs = [get_best_epoch(history, best_metric) for history in histories]

    model_optimal_values = [best_epoch['metrics'][best_metric] for best_epoch in best_epochs]
    indices = np.argsort(model_optimal_values)

    ordered_models = [{'params': params[i], 'history': histories[i], 'best_epoch': best_epochs[i]}
                      for i in indices]
    return ordered_models


def report_model_results(loaded_model_results, best_metrics=['epi_abs', 'epi_sqr', 'ssd', 'sed'], num_best_models=5):
    """
    Reports (prints) the results of the given loaded models.
    The params and best metric values of the num_best_models best model are printed and the histories of the best
    model is plotted.

    :param loaded_model_results: List of model results. Each item in the list is a
        dict with entries "params", "history" and "history_details" for each of the loaded files.
    :param best_metrics: List of metric names used to order the results. For each of the metrics the results
        are reported.
    :param num_best_models: Number of best models for which results are reported.
    :return: All ordered models. Dict where each item represents the ordered models for a given metric. (key = metric).
        Each ordered model is a list of dicts with entries "params", "history" and "best_epoch". Each dict in the list represents
        the results of a model. The results are ordered starting with the best model (lowest metric values).
        - params: Dict of params of the model.
        - history: History dict where each item is a list of floats.
        - best_epoch: Dict with entries "epoch" which contains the 1-based index of the best epoch
            and "metrics" which is a dict containing the metric values of the best epoch.
    """
    all_ordered_models = {}
    for best_metric in best_metrics:
        ordered_models = get_ordered_models(loaded_model_results, best_metric)
        all_ordered_models[best_metric] = ordered_models

        best_model = ordered_models[0]
        best_model_epoch = best_model['best_epoch']
        print('==================== Results for metric "%s" ====================' % best_metric)
        print('-------------------- Results of best model --------------------')
        report_model_metrics(best_model)

        print('\n\n-------------------- Parameters of best model --------------------')
        report_model_params(best_model)

        print('\n\n-------------------- Histories of best model--------------------')
        plot_metrics(best_model['history'])

        num_best_models = min(num_best_models, len(ordered_models)-1)
        print('\n\n-------------------- Next %d best models--------------------' % num_best_models)
        for i in range(num_best_models):
            print('\n-----> Best model %d <-----' % (i+2))
            model = ordered_models[i+1]
            print('--- Results ---')
            report_model_metrics(model)
            print('\n--- Parameters ---')
            report_model_params(model)
        print('==================== End results ====================\n\n\n')
    return all_ordered_models


def convert_model_results_to_csvs(loaded_model_results, csv_file_path_prefix, best_metrics):
    """
    Saves the model results of all loaded models in multiple csv files which can be used to compare the models.
    One csv file for each of the given metrics is created. The metric is used to find the best epoch.
    Each csv file contains columns for all parameters of each model, the best model epoch and the metrics
    of the best epoch of each model.
    :param loaded_model_results: List of loaded results.
        List of dicts with entries "params", "history" and "history_details".
        Each list item represents a single model.
    :param csv_file_path_prefix: Predix to the path of the csv file which is created.
    :param best_metrics: List of metric names used to find the best epoch of each model
        for which the metrics are included in the csv files.
    """
    for metric in best_metrics:
        csv_file_path = csv_file_path_prefix + '_' + metric + '.csv'
        convert_model_results_to_csv(loaded_model_results, csv_file_path, metric)


def convert_model_results_to_csv(loaded_model_results, csv_file_path, best_metric):
    """
    Saves the model results of all loaded models in a csv file which can be used to compare the models.
    The csv file contains columns for all parameters of each model, the best model epoch and the metrics
    of the best epoch of each model.

    :param loaded_model_results: List of loaded results.
        List of dicts with entries "params", "history" and "history_details".
        Each list item represents a single model.
    :param csv_file_path: Path of the csv file which is created.
    :param best_metric: Metric name used to find the best epoch of each model for which the metrics are included
         in the csv file.
    """
    print('Creating results csv for best metric %s' % best_metric)
    ignored_model_results = [
        result['params']['model_name'] + '_' + result['params']['timestamp'] + '_' + str(result['params']['version'])
        for result in loaded_model_results if best_metric not in result['history']]
    if len(ignored_model_results) > 0:
        print('The following models will be ignored as they do not have this metric: %s' % ignored_model_results)

    loaded_model_results = [result for result in loaded_model_results if best_metric in result['history']]
    params = [result['params'] for result in loaded_model_results]
    histories = [result['history'] for result in loaded_model_results]
    best_epochs = [get_best_epoch(history, best_metric) for history in histories]
    model_version_names = [param['model_name'] + '_' + param['timestamp'] + '_' + str(param['version'])
                           for param in params]

    param_names_set = set()
    for model_params in params:
        param_names_set |= set(model_params.keys())
    param_names = list(param_names_set)
    param_names.sort()
    metric_names_set = set()
    for model_history in histories:
        metric_names_set |= set(model_history.keys())
    metric_names = list(metric_names_set)
    metric_names.sort()
    field_names = ['#model'] + param_names + ['best_epoch'] + metric_names

    with open(csv_file_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, field_names)
        writer.writeheader()

        for (param, best_epoch, model_version_name) in zip(params, best_epochs, model_version_names):
            row_dict = dict(**param, best_epoch=best_epoch['epoch'], **best_epoch['metrics'])
            row_dict['#model'] = model_version_name
            writer.writerow(row_dict)


def save_result_plots(loaded_model_results, output_folder):
    """
    Plots the histories of the given model results and saves them into files.
    One file for each metric and each model is created.

    :param loaded_model_results: List of loaded results.
        List of dicts with entries "params", "history" and "history_details".
        Each list item represents a single model.
    :param output_folder: Folder into which the plot pdf files are stored.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    params = [result['params'] for result in loaded_model_results]
    histories = [result['history'] for result in loaded_model_results]

    for (param, history) in zip(params, histories):
        file_prefix = output_folder + '/' + param['model_name'] + '_' + param['timestamp'] + '_' + str(param['version'])
        plot_metrics_to_files(history, file_prefix)


def extract_all_model_results(target_folder, base_folder=None):
    """
    Extracts all the models results of all currently saved models (in the models folder) into the target folder.

    The result json files are all directly saved in the target folder with name suffixes.
    Images are copied to an extra folder.
    Weights and regression inputs are not extracted.

    :param target_folder: Folder into which the results are extracted.
    :param base_folder: Base folder of all models. If None the default folder is used.
    """
    if base_folder is None:
        base_folder = MODELS_FOLDER

    for model_name in os.listdir(base_folder):
        for timestamp in os.listdir(base_folder + '/' + model_name):
            for params_version in os.listdir(base_folder + '/' + model_name + '/' + timestamp):
                extract_model_results(model_name, timestamp, params_version, target_folder, base_folder=base_folder)


def extract_model_results(model_name, timestamp, params_version, target_folder, base_folder=None):
    """
    Extracts the model results of a single model.

    Only results json and images are extracted. Weights and regression inputs are not extracted.

    :param model_name: Name of the root model.
    :param timestamp: Timestamp of the model.
    :param params_version: Version of the model (from hyperparam tuning).
    :param target_folder: Folder into which the results are extracted.
    :param base_folder: Base folder of all models. If None the default folder is used.
    """
    folder = get_model_folder(model_name, timestamp, params_version, base_folder=base_folder)
    combined_name = model_name + '_' + timestamp + '_' + params_version
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Extract results file
    new_results_file = target_folder + '/model_results__' + combined_name + '.json'
    shutil.copy(folder + '/' + RESULTS_FILE_NAME, new_results_file)

    # Extract image results
    img_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    if len(img_files) > 0:
        images_folder = target_folder + '/model_images/' + combined_name
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        for img_file in img_files:
            shutil.copy(folder + '/' + img_file, images_folder)


class NumpyJsonEncoder(json.JSONEncoder):
    """
    Json encoder for numpy types
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


#######################################################################
def add_params_to_models(model_name, added_params, base_folder=None):
    """
    Utility method to modify the parameters of stored model results.

    This can e.g. be used, when new hyperparams where added to the model after an old model
    has already been trained.
    The given hyperparams are added to all models of a given root model. Already present values cannot be overwritten
    an error will occur in this case.
    The json files are edited by this function.

    :param model_name: Name of the root model.
    :param added_params: Dict containing the added params.
    :param base_folder: Base folder of all models. If None the default folder is used.
    """
    model_base_folder = _get_model_base_folder(model_name, base_folder=base_folder)
    for timestamp in os.listdir(model_base_folder):
        for params_version in os.listdir(model_base_folder + '/' + timestamp):
            results = load_param_model_results(model_name, timestamp, params_version, base_folder=base_folder)
            old_params = results['params']
            new_params = dict(list(old_params.items()) + list(added_params.items()))
            folder = model_base_folder + '/' + timestamp + '/' + params_version
            save_results(folder, new_params, dict(history=results['history'], history_details=results['history_details']))

