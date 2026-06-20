import json
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0+PTX"  # per nuove GPU

import itertools
import sys
from datetime import datetime

from src.config import Parameters
from src.dataset.utils import setup_seed
from src.training.train import train, test
from src.training.training_module import Training
from src.utils.utils import (get_model, get_datamodule,
                             set_params_wrt_dataset, update_run_metrics, update_seed_metrics, initialize_log_parameters)


def build_combinations(search_space: dict) -> list:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_single_seed(combo: dict,
                    run_params: Parameters) -> tuple:
    if run_params.reproducible:
        setup_seed(run_params.seed)

    # Data
    data_module_instance, run_params = get_datamodule(run_params)

    # Model
    training_module = Training(run_params)

    # Train
    train(training_module, data_module_instance, run_params)

    # Test
    res_test = test(training_module, data_module_instance, run_params)
    return training_module, res_test, run_params


def run_single_combination(combo: dict,
                           cont: int,
                           global_config: dict,
                           seed_list: list):
    grid_params_dict = initialize_log_parameters(cont, combo)
    val_results, test_results = [], []

    for seed in seed_list:
        global_config['seed'] = seed
        run_params = Parameters().update(combo|global_config)

        train_module, res_test, run_params = run_single_seed(combo, run_params)
        val_results, test_results = update_seed_metrics(train_module, res_test, val_results, test_results)

    update_run_metrics(val_results, test_results, grid_params_dict, run_params)


def main():
    search_space = {
        'dataset_name': ['activity'],  # 'activity', 'ushcn', 'mimic', 'physionet'
        'dropout': [0.0],
        'lr': [1e-3],
        'batch_size': [32],
        'gnn_name': ['GAT'],
        'hid_dim': [32],
        'gnn_layers': [2, 1],
        'pool_num_heads': [16, 8, 4],
        'inner_mode': [4, 3, 2, 1],
        'decoder_name' : ['simple', ]  #  'INR', 'film', 'gated', 'gru', 'filmSwiglu', 'simple', 'crossAttn'
    }


    global_config = {
        'save_ckpts': False,
        'early_stop_callback_flag': True,
        'save_logs': True,
        'reproducible': True}

    seed_list = [654, 897, 26, ]
    log_folder = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    global_config['logs_dir'] = os.path.join(Parameters().logs_dir, log_folder)
    os.makedirs(global_config['logs_dir'], exist_ok=True)
    combinations = build_combinations(search_space)
    print(f'Total combinations: {len(combinations)}')

    # Save serach params
    static_run_params = Parameters().update(combinations[0] | global_config)
    static_run_params.to_yaml(os.path.join(global_config['logs_dir'], 'static_run_params.yaml'))
    with open(os.path.join(global_config['logs_dir'], 'search_space_params.json'), "w", encoding="utf-8") as f:
        json.dump(search_space|global_config, f, indent=4, ensure_ascii=False)

    last_run=0
    for cont, combo in enumerate(combinations):
        if cont < last_run:
            continue
        # try:
        #     print(f'\nRun {cont + 1}/{len(combinations)}')
        #     run_single_combination(combo, cont, global_config, seed_list)
        # except:
        #     print('Error: ', sys.exc_info()[0])
        print(f'\nRun {cont + 1}/{len(combinations)}')
        run_single_combination(combo, cont, global_config, seed_list)

if __name__ == '__main__':
    main()
