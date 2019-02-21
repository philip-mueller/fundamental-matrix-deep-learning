default_params = {
    'discriminator_classifier_neurons': [1024, 512],
    'discriminator_channels': [64, 128, 256, 512],
    'discr_iterations': 1,
    'freeze_discriminator': False,

    'generator_channels': [64, 128, 256, 512],
    'generator_skip_connections': True,
    'generator_bottleneck_weight_reg': None,
    'generator_bottleneck_activity_reg': None,
    'bottleneck_size': 128,

    'regressor_dense_neurons': [1024, 512],
    'regressor_channels': [64, 128, 256, 512, 8],
    'regressor_batchnorm': True,
    'regressor_bottleneck_batchnorm': True,
    'regressor_derived_features_batchnorm': True,
    'regressor_train_iterations': None,
    'regressor_batch_size': 32,
    'lr_R_decay': None,

    'generator_loss_weights': [0.5, 0.5],
    'dropout': 0.75,

    'reconstruction_type': 'reconstruct',
    'norm': 'fro'
}

redirect_params = {
    'generator_dropout': 'dropout',
    'regressor_dropout': 'dropout',
    'discriminator_dropout': 'dropout',
    'sample_train_epochs': 'sample_epochs',
    'sample_predict_epochs': 'sample_epochs',
    'train_discr_iterations': 'discr_iterations',
    'predict_discr_iterations': 'discr_iterations',

    'lr_R': 'lr_G'
}


class Params:
    """
    Helper for parameters which uses redirect or default params when no param is defined for a given key.

    Behaves like a normal dictionary but includes handling for default values.
    """
    def __init__(self, params_dict):
        self.params = params_dict

    def __getitem__(self, item):
        """
        If in the params_dict the requested key is not defined then first the redirect_params are checked.
        if there the key is defined the value for that key in redirect_params is used as new key
        to get the value from a Params object (the key is redirected to another key).
        If also nor redirect key is specified,
        the default values are checked and the corresponding default value is returned.
        If also there no values is found a ValueError is raised.
        :param item: Key for the item.
        :return: Found value.
        """
        if item in self.params:
            return self.params[item]
        elif item in redirect_params:
            redirected_params = redirect_params[item]
            return self[redirected_params]
        elif item in default_params:
            return default_params[item]
        else:
            raise ValueError('Parameter %s is not defined and does not have a default value' % item)

    def __setitem__(self, key, value):
        self.params[key] = value

    def __contains__(self, item):
        return item in self.params \
               or (item in redirect_params and redirect_params[item] in self) \
               or item in default_params

    def keys(self):
        all_keys = set(self.params.keys()) | set(default_params.keys()) | set(redirect_params.keys())
        all_keys = {key for key in all_keys if (key in self)}
        return all_keys

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def to_dict(self):
        """
        Converts this object to a real dict that contains all values and default values for keys which are
        not defined.
        :return: Dictionary
        """
        return dict(self.items())
