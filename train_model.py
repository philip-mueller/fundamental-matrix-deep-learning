import tkinter
from train.model.model_trainers import FMatrixGanTrainer
from train.utils.metrics import *
from dataset_utils.dataset_loading import load_FMatrix_data
from train.utils.hyperparameters import tune_hyperparameters

def train_model(model_folder, train, val, params, **kwargs):
    model = FMatrixGanTrainer(params, model_folder, metrics=[epi_abs, epi_sqr, ssd, sed])
    history = model.fit(train, val, save_imgs=True, **kwargs)
    return history, model

#p = {
#    'lr_D': 7e-06,
#    'lr_G': 9e-06,
#    'lr_R': 3.0e-05, 
#    'lr_R_decay': 0.96,
#    'derived_feature_layers': None,
#    'use_images': False,
#    'sample_epochs': 150,
#    'regressor_epochs': 50,
#    'norm': 'fro',
#    'dataset': 'synthetic_horse_rotation',
#    'train_method': 'separated',
#}
#model_name = 'bottleneck_only'

#p = {
#    'lr_D': 7e-06,
#    'lr_G': 9e-06,
#    'lr_R': 3.0e-05, 
#    'lr_R_decay': 0.96,
#    'derived_feature_layers': [[0, 1, 2, 3]],
#    'use_images': False,
#    'sample_epochs': 150,
#    'regressor_epochs': 50,
#    'norm': 'fro',
#    'dataset': 'synthetic_horse_rotation',
#    'train_method': 'separated',
#}
#model_name = 'derived_features'

p = {
    'lr_D': 7e-06,
    'lr_G': 9e-06,
    'lr_R': 2.2e-05, 
    'lr_R_decay': 0.95,
    'derived_feature_layers': [[0, 1, 2, 3]],
    'use_images': True,
    'sample_epochs': 150,
    'regressor_epochs': 50,
    'norm': 'fro',
    'dataset': 'synthetic_horse_rotation',
    'train_method': 'separated',
}
model_name = 'derived_features_imgs'

tune_hyperparameters(train_model, load_FMatrix_data, model_name, p, 
                     validate_interval=1, 
                     save_weights_interval=1, 
                     sample_plot_interval=100, verbose=1)

