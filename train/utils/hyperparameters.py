import numpy as np
from datetime import datetime
import itertools
import os
import gc
from tensorflow.python.keras import backend as K

from train.utils.params import Params
from train.utils.results import get_model_folder, save_results


def generate_parameter_combinations(params_def):
    """
    Generates all possible hyperparameter combinations using params_def.

    :param params_def: Definition of hyperparameter ranges.
    :return: List of dictionaries. Each dict contains the hyperparams of a single configuration.
    """
    param_lists = []
    for param_name, param_def in iter(params_def.items()):
        param_values = generate_values_for_param(param_def)
        param_lists.append([(param_name, param_value) for param_value in param_values])

    for param_combination in itertools.product(*param_lists):
        yield {param[0]: param[1] for param in param_combination}


def generate_values_for_param(param_def, end_inclusive=True, default_range=(None, None), default_value=None):
    """
    Generates possible hyperparameter values for a single hyperparam range.

    :param param_def: Parameter definition, either single value, list or tuple (for ranges)
    :param end_inclusive: True, if the end should be inclusive for ranges.
    :param default_range: Default range which is used when param_def is range tuple,
        but not all range values (e.g. start or end) are defined.
    :param default_value: Default value when param_def is None
    :return: List of possible parameters.
    """
    if isinstance(param_def, tuple):
        if len(param_def) >= 4:
            start, end, num, range_type = param_def
        else:
            range_type = 'lin'
            start, end, num = param_def
        
        if start is None:
            start = default_range[0]
        if end is None:
            end = default_range[1]

        if range_type == 'log':
            param_values = np.logspace(np.log10(start), np.log10(end), num=num, base=10, endpoint=end_inclusive)
        elif range_type == 'lin':
            param_values = np.linspace(start, end, num=num, endpoint=end_inclusive)
        elif range_type == 'rand':
            param_values = np.random.uniform(start, end, num)
        else:
            raise ValueError('Unkown range type %s' % range_type)
    elif isinstance(param_def, list):
        param_values = param_def
    elif param_def is None:
        param_values = [default_value]
    else:
        param_values = [param_def]
    return param_values


def tune_hyperparameters(training_fn, data_loader, model_name, params_def, **kwargs):
    """
    Runs the training_fn with all possible hyperparameter combinations defined by params_def and stores the results.

    The results are stored in the models folder.
    Each model_name gets its own folder, within that one folder for each hyperparameter tuning is created
    using the current timestamp. Within this folder each hyperparam configuration gets it own folder.
    In this folder all data, which the model creates is stored as well as a results.json,
    which contains the used hyperparameter and the training history of the model.

    :param training_fn: Training function:
        results, model = training_fn(model_folder, train, val, params, **kwargs)
        results is a dict with keys 'history' and 'history_details', which contain the main training
        history (a dict of lists each representing a metric) and detailed histories (which may be None)
    :param data_loader: Data loading function to load data for the traiing:
        train, val = data_loader(params, **kwargs)
        train and val are both DataLoaders, params is a single hyperparam configuration.
    :param model_name: Name of the model
    :param params_def: Hyperparameter definitions.
    :param kwargs: Optional args passed to data loader and training fn
    :return: model_name, timestamp which can be used to load the results of the trained models.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
    param_combinations = list(generate_parameter_combinations(params_def))

    print('Starting hyperparameter tuning. Trying %d combinations of parameters...' % len(param_combinations))

    for i, params in enumerate(param_combinations):
        params = Params(params)
        version = i+1
        params['model_name'] = model_name
        params['timestamp'] = timestamp
        params['version'] = version

        model_folder = get_model_folder(model_name, timestamp, version)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        print('==================== Training model #%d ====================' % version)
        for param_name, param_value in iter(params.items()):
            print('> %s = %s' % (param_name, param_value))
        print('============================================================')

        print('Loading data...')
        train, val = data_loader(params, **kwargs)
        print('Loading data done.')

        print('Start training...')
        start_time = datetime.now()
        results, model = training_fn(model_folder, train, val, params, **kwargs)
        end_time = datetime.now()
        elapsed = str(end_time - start_time)
        params['training_time'] = elapsed
        print('Training done. Training took: %s' % elapsed)

        print('Writing results...')
        save_results(model_folder, params, results)
        print('Done.')

        # Free resources
        del model
        gc.collect()
        K.clear_session()

    return model_name, timestamp
