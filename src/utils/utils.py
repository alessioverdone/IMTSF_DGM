import os

import torch
import math

from src.layers.dgm import DGM_module
from src.layers.hipatch import Hi_Patch
from src.dataset.parse_datasets import parse_datasets
from src.layers.moe import MixtureOfExpertsDGM
from src.layers.grape import Grape, GrapeDgm


def get_model(hparams):
    if 'dgm' in hparams.model:
        if hparams.dgm_mode == 'simple':
            model = DGM_module(hparams)
        elif hparams.dgm_mode == 'moe':
            model = MixtureOfExpertsDGM(hparams,
                                       hparams.num_nodes,
                                       hparams.node_features,
                                       hparams.prediction_window,
                                       hparams.edge_index,
                                       edge_weight=None,
                                       hidden=hparams.emb_dim)
    elif hparams.model == 'hi-patch':
        model = Hi_Patch(hparams)
    elif hparams.model == 'grape':
        model = Grape(hparams)
        # model = GrapeDgm(hparams)
    else:
        raise Exception('Error in select the model!')

    model = model.to(hparams.device)
    return model


def set_params_wrt_dataset(run_params, dataModuleInstance):
    ### Model setting DGM ###
    if run_params.model == 'dgm':
        # Configure input feature size
        if run_params.pre_fc is None or len(run_params.pre_fc) == 0:
            if len(run_params.dgm_layers[0]) > 0:
                run_params.dgm_layers[0][0] = dataModuleInstance.train_data.n_features
            run_params.conv_layers[0][0] = dataModuleInstance.train_data.n_features
        else:
            if run_params.dataset_name in ['PV', 'wind']:
                run_params.pre_fc[0] = dataModuleInstance.train_data.dataset.n_features
            elif run_params.dataset_name in ['METR-LA', 'solar', 'electricity']:
                run_params.pre_fc[0] = run_params.lags
            else:
                raise ValueError('Define dataset name correctly!')

        run_params.fc_layers[-1] = run_params.prediction_window
        run_params.node_features = run_params.lags
        run_params.edge_index  = dataModuleInstance.torch_dataset.edge_index  # TODO: Con PV o Wind dà errore!

    ### Model setting Hi-Patch ###
    elif run_params.model == 'hi-patch' or run_params.model == 'grape':
        # Hi-patch/grape parameters
        run_params.ndim = dataModuleInstance["input_dim"]
        run_params.npatch = int(math.ceil((run_params.history - run_params.patch_size) / run_params.stride)) + 1
        run_params.patch_layer = layer_of_patches(run_params.npatch)
        run_params.scale_patch_size = run_params.patch_size / (run_params.history + run_params.pred_window)
        run_params.task = 'forecasting'

        # DGM params update
        run_params.dgm_layers[0][0] = run_params.hid_dim
        run_params.conv_layers[0][0] = run_params.hid_dim
        run_params.pre_fc[0] = run_params.hid_dim
        run_params.pre_fc[-1] = run_params.hid_dim

    else:
        raise ValueError('Define model name correctly!')

    return run_params


# def make_dgm_network_parameters(emb_dim):
#     cost = emb_dim / 32
#     conv_layers = [[emb_dim, emb_dim], [emb_dim, int(emb_dim / 2)], [int(emb_dim / 2), int(emb_dim / 4)]]
#     dgm_layers = [[emb_dim, int(emb_dim / 2), int(emb_dim / 8)], [int(36 * cost), int(emb_dim / 2), int(emb_dim / 8)],
#                   []]
#     fc_layers = [int(8 * cost), int(8 * cost), int(3 * cost)]
#     pre_fc = [-1, emb_dim]
#     return conv_layers, dgm_layers, fc_layers, pre_fc

# Recursive function to determine patch layers
def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    else:
        return layer_of_patches(n_patch + 1)


def make_dgm_network_parameters_v2(emb_dim):
    pre_fc = [-1, emb_dim]

    dgm_layers = [[emb_dim, int(emb_dim / 2)], [emb_dim, int(emb_dim / 2)], []]
    conv_layers = [[emb_dim, int(emb_dim / 2)], [int(emb_dim / 2), int(emb_dim / 2)],
                   [int(emb_dim / 2), int(emb_dim / 4)]]

    fc_layers = [int(emb_dim / 4), -1]
    return conv_layers, dgm_layers, fc_layers, pre_fc


def initialize_parameters_old(cont, run_combination):



    model_item = run_combination[0]
    dataset_name_item = run_combination[1]
    emb_dim_item = run_combination[2]
    k_item = run_combination[3]
    batch_size_item = run_combination[4]
    lags_item = run_combination[5]
    prediction_window_item = run_combination[6]
    dropout_item = run_combination[7]
    grid_params_name = ['Run',
                        'model',
                        'dataset_name',
                        'emb_dim',
                        'k',
                        'batch_size',
                        'lags',
                        'prediction_window',
                        'dropout',
                        'val_mse_mean',
                        'val_mse_std',
                        'val_rmse_mean',
                        'val_rmse_std',
                        'val_mae_mean',
                        'val_mae_std',
                        'val_mape_mean',
                        'val_mape_std',
                        'test_mse_mean',
                        'test_mse_std',
                        'test_rmse_mean',
                        'test_rmse_std',
                        'test_mae_mean',
                        'test_mae_std',
                        'test_mape_mean',
                        'test_mape_std']
    grid_params = [cont,
                   model_item,
                   dataset_name_item,
                   emb_dim_item,
                   k_item,
                   batch_size_item,
                   lags_item,
                   prediction_window_item,
                   dropout_item,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.]
    grid_params_dict = dict(zip(grid_params_name, grid_params))
    output_string = ' '.join([f'{name}: {value}' for name, value in grid_params_dict.items()])
    print(output_string)
    return grid_params_dict



def initialize_log_parameters(cont: int, combo: dict) -> dict:
    METRICS = ['mse', 'rmse', 'mae']
    SPLITS = ['val', 'test']

    # colonne metriche: val_mse_mean, val_mse_std, ...
    metric_keys = [f'{split}_{metric}_{stat}'
                   for split in SPLITS
                   for metric in METRICS
                   for stat in ('mean', 'std')]

    grid_params = {'Run': cont, **combo, **{k: 0. for k in metric_keys}}

    print(' '.join(f'{k}: {v}' for k, v in grid_params.items()))
    return grid_params




def get_datamodule(run_params):
    # Parse dataset and initialize model
    if run_params.dataset_name in ["physionet", "mimic", "ushcn", "activity"]:
        data_module_instance = parse_datasets(run_params,
                                              run_params.patch_ts)
    else:
        raise ValueError('Define dataset name correct!')

    run_params = set_params_wrt_dataset(run_params, data_module_instance)  #TODO: da adattare al modello

    return data_module_instance, run_params


def update_seed_metrics(model, res_test, val_results, test_results):
    best_val_mse, best_val_rmse, best_val_mae = model.best_mse, model.best_rmse, model.best_mae

    # Testing
    test_mse = res_test[0]['test_mse']
    test_rmse = res_test[0]['test_rmse']
    test_mae = res_test[0]['test_mae']

    val_results.append([best_val_mse, best_val_rmse, best_val_mae])
    test_results.append([test_mse, test_rmse, test_mae])

    print(f'best_val_mse: {best_val_mse}')
    print(f'best_val_rmse: {best_val_rmse}')
    print(f'best_val_mae: {best_val_mae}')
    print(f'test_mse: {test_mse}')
    print(f'test_rmse {test_rmse}')
    print(f'test_mae: {test_mae}')
    return val_results, test_results



def update_run_metrics(val_results,
                       test_results,
                       grid_params_dict,
                       run_params):
    metrics = ['mse', 'rmse', 'mae']
    splits = ['val', 'test']
    results = {'val':  torch.tensor(val_results),
               'test': torch.tensor(test_results)}

    grid_params_dict.update({f'{split}_{metric}_{stat}': float(getattr(torch, stat)(results[split][:, i]))
                            for split in splits
                            for i, metric in enumerate(metrics)
                            for stat in ('mean', 'std')})

    print(' '.join(f'{k}: {v}' for k, v in grid_params_dict.items()))
    output_string = ' '.join([f'{k}: {v}' for k, v in grid_params_dict.items()])

    if run_params.save_logs:
        os.makedirs(run_params.logs_dir, exist_ok=True)
        with open(os.path.join(run_params.logs_dir, 'log.txt'), 'a') as file:
            print(output_string, file=file)


# def update_run_metrics_old(val_results, test_results, grid_params_dict, run_params):
#     val_results = torch.tensor(val_results)
#     test_results = torch.tensor(test_results)
#     val_mse_over_seeds = val_results[:, 0]
#     val_rmse_over_seeds = val_results[:, 1]
#     val_mae_over_seeds = val_results[:, 2]
#     val_mape_over_seeds = val_results[:, 3]
#
#     test_mse_over_seeds = test_results[:, 0]
#     test_rmse_over_seeds = test_results[:, 1]
#     test_mae_over_seeds = test_results[:, 2]
#     test_mape_over_seeds = test_results[:, 3]
#
#     grid_params_dict.update({
#         'val_mse_mean': float(torch.mean(val_mse_over_seeds)),
#         'val_mse_std': float(torch.std(val_mse_over_seeds)),
#         'val_rmse_mean': float(torch.mean(val_rmse_over_seeds)),
#         'val_rmse_std': float(torch.std(val_rmse_over_seeds)),
#         'val_mae_mean': float(torch.mean(val_mae_over_seeds)),
#         'val_mae_std': float(torch.std(val_mae_over_seeds)),
#         'val_mape_mean': float(torch.mean(val_mape_over_seeds)),
#         'val_mape_std': float(torch.std(val_mape_over_seeds)),
#         'test_mse_mean': float(torch.mean(test_mse_over_seeds)),
#         'test_mse_std': float(torch.std(test_mse_over_seeds)),
#         'test_rmse_mean': float(torch.mean(test_rmse_over_seeds)),
#         'test_rmse_std': float(torch.std(test_rmse_over_seeds)),
#         'test_mae_mean': float(torch.mean(test_mae_over_seeds)),
#         'test_mae_std': float(torch.std(test_mae_over_seeds)),
#         'test_mape_mean': float(torch.mean(test_mape_over_seeds)),
#         'test_mape_std': float(torch.std(test_mape_over_seeds))
#     })
#     output_string = ' '.join([f'{name}: {value}' for name, value in grid_params_dict.items()])
#
#     if run_params.save_logs:
#         with open(f'../logs/logs_{run_params.dataset_name}_mean_std_{run_params.lags}_{run_params.prediction_window}_{run_params.dgm_mode}.txt', 'a') as file:
#             print(output_string, file=file)
