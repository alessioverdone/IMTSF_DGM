import os
from pathlib import Path
from pprint import pformat
from random import SystemRandom

import numpy as np
import torch
import yaml


class Parameters:
    def __init__(self):
        project_dir = Path(__file__).resolve().parents[1]

        # Trainer parameters
        self.accelerator = 'gpu'
        self.log_every_n_steps = 300
        self.max_epochs = 5
        self.enable_progress_bar = True
        self.check_val_every_n_epoch = 3
        self.node_features = 23
        self.time_series_step = 4
        self.device = torch.device('cuda' if self.accelerator == 'gpu' and torch.cuda.is_available() else 'cpu')

        # Datasets
        self.dataset_name = 'physionet'  # ['PV', 'wind', 'METR-LA', 'Electricity', 'solar', 'physionet']
        self.data_dir = os.path.join(project_dir, 'data')
        self.dataset_path = os.path.join(project_dir,
                                         'data',
                                         'Generated_time_series_output_31_with_weigth_multivariate_and_time.json')
        self.config_path = os.path.join(project_dir,
                                         'registry',
                                         'configurations')

        # DGM Model parameters
        self.conv_layers = [[32, 32], [32, 16], [16, 8]]
        self.dgm_layers = [[32, 16, 4], [36, 16, 4], []]
        self.fc_layers = [8, 8, 3]
        self.pre_fc = [-1, 32]
        self.emb_dim = 64
        self.gfun = 'gcn'
        self.ffun = 'gcn'  # 'gcn', 'gat', None, 'mlp', 'knn'
        self.k = 5
        self.pooling = 'add'
        self.distance = 'euclidean'
        self.dgm_mode = 'simple'

        self.dropout = 0.0
        self.lr = 1e-3
        self.test_eval = 10
        self.model = 'prova'  # 'dgm_gcn', 'dgm_gat', 'gcn', 'gat', 'mlp' 'hi-patch'

        self.early_stop_callback_flag = False
        self.early_stop_patience = 4

        self.lags = 24
        self.prediction_window = 24
        # self.batch_size = 128
        self.num_nodes = 31
        self.logging = False
        self.save_ckpts = False
        self.save_logs = True
        self.chkpt_dir = ''
        self.reproducible = True
        self.seed = 42

        self.connectivity_threshold_tsl = 0.1
        self.train_len = 0.6
        self.val_len = 0.2
        self.test_len = 0.2
        self.workers = 2
        self.stride_tsl = 1

        # Hi-Patch parameters
        self.state = 'def'
        self.epoch = 1000
        self.patience = 10
        self.history = 24
        self.pred_window = 1
        self.n = int(1e8)  #n_samples
        self.n_months = 48
        self.ndim = 0
        self.patch_size = 24  # 24
        self.npatch = 1
        self.patch_layer = 1
        self.scale_patch_size = 1
        self.task = 'forecasting'
        self.hid_dim = 32
        self.N = 1
        self.logmode = 'a'
        self.lr = 1e-3
        self.w_decay = 0.0
        self.batch_size = 4
        self.load = None
        self.seed = 1
        # self.dataset = 'physionet'
        self.quantization = 0.0
        # self.model = 'Hi-Patch'
        self.nhead = 1
        self.nlayer = 3
        self.ps = 24
        self.stride = 24
        self.alpha = 1
        self.res = 1
        self.gpu = '0'
        self.npatch = int(np.ceil((self.history - self.patch_size) / self.stride)) + 1
        self.PID = os.getpid()
        self.experimentID = -1
        self.patch_ts = True  # problemi con observed_tp, non vengono esplicitati i timestep per canale a differenza di patch_ts = True



    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(vars(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'Parameters':
        instance = cls()
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            setattr(instance, key, value)
        return instance

    def __str__(self) -> str:
        # pformat rende leggibili liste/nidificazioni
        body = pformat(vars(self), sort_dicts=False, width=120)
        return f"{self.__class__.__name__}(\n{body}\n)"


def initialize_configuration(config_file=None):
    run_params = Parameters()
    if config_file is not None:
        config_path = os.path.join(run_params.config_path,
                                   config_file)
        run_params = Parameters.from_yaml(config_path)
        print(f'Loaded configuration named: {config_file}')
    else:
        run_params.experimentID = int(SystemRandom().random() * 100000)
        print(f'Loaded default configuration.')
    print('List of parameters:')
    print(run_params)
    return run_params