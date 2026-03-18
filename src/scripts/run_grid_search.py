import itertools
import random

import numpy as np
import torch

from src.config import Parameters
from src.training.train import train, test
from src.utils.utils import ( get_model, initialize_parameters, get_datamodule,
                                   set_params_wrt_dataset, update_current_configuration,
                                   update_run_metrics, update_seed_metrics)

# --- Search space -----------------------------------------------------------

SEARCH_SPACE = {
    'f_models':           ['gat', 'gcn1d', 'gcn'],
    'g_models':           ['gat', 'gcn1d', 'gcn'],
    'datasets':           ['METR-LA'],
    'emb_dims':           [32, 64],
    'k_values':           [2, 3, 4],
    'batch_sizes':        [8, 16, 32],
    'lags':               [12],
    'prediction_windows': [12],
    'dropouts':           [0.2, 0.1, 0.0],
}

GLOBAL_CONFIG = {
    'save_ckpts':               False,
    'early_stop_callback_flag': True,
    'save_logs':                True,
    'reproducible':             True,
    'seed_list':                [253, 3667, 8],
}

# ---------------------------------------------------------------------------


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_combinations() -> list:
    dgm_models = [f'dgm_{f}_{g}'
                  for f in SEARCH_SPACE['f_models']
                  for g in SEARCH_SPACE['g_models']]
    return list(itertools.product(
        dgm_models,
        SEARCH_SPACE['datasets'],
        SEARCH_SPACE['emb_dims'],
        SEARCH_SPACE['k_values'],
        SEARCH_SPACE['batch_sizes'],
        SEARCH_SPACE['lags'],
        SEARCH_SPACE['prediction_windows'],
        SEARCH_SPACE['dropouts'],
    ))


def run_single_seed(combo: tuple, seed: int, device: torch.device) -> tuple:
    run_params = Parameters()
    run_params = update_current_configuration(
        combo, seed, run_params,
        GLOBAL_CONFIG['reproducible'],
        GLOBAL_CONFIG['save_ckpts'],
        GLOBAL_CONFIG['save_logs'],
        GLOBAL_CONFIG['early_stop_callback_flag'],
    )

    if run_params.reproducible:
        set_seed(seed)

    dataModuleInstance, run_params = get_datamodule(run_params)
    run_params = set_params_wrt_dataset(run_params, dataModuleInstance)

    model = get_model(run_params).to(device)

    train(model, dataModuleInstance, run_params, device)
    res_test = test(model, dataModuleInstance, device)

    return model, res_test, run_params


def run_single_combination(combo: tuple, cont: int, device: torch.device):
    grid_params_dict = initialize_parameters(cont, combo)
    val_results, test_results = [], []
    run_params = None

    for seed in GLOBAL_CONFIG['seed_list']:
        print(f'  [seed={seed}] ffun={combo[0].split("_")[-2]}  gfun={combo[0].split("_")[-1]}')
        model, res_test, run_params = run_single_seed(combo, seed, device)
        val_results, test_results = update_seed_metrics(model, res_test, val_results, test_results)

    update_run_metrics(val_results, test_results, grid_params_dict, run_params)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combinations = build_combinations()
    print(f'Total combinations: {len(combinations)} | Device: {device}')

    for cont, combo in enumerate(combinations):
        model_name, dataset, emb, k, bs = combo[0], combo[1], combo[2], combo[3], combo[4]
        print(f'\nRun {cont + 1}/{len(combinations)} | {model_name} | {dataset} | emb={emb} k={k} bs={bs}')
        run_single_combination(combo, cont, device)


if __name__ == '__main__':
    main()
