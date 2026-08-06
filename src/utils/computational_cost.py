"""
Benchmark di costo computazionale + metriche di task.

Esegue uno sweep (prodotto cartesiano di uno search space, come run_grid_search.py)
e per ogni run (combo x seed) colleziona:
  - metriche di task  : val/test mse, rmse, mae
  - metriche di tempo : tempo totale, per epoca, per batch, throughput, latenza inferenza
  - metriche di modello: numero parametri, dimensione in MB, stato ottimizzatore
  - metriche di memoria: picco VRAM (train / inferenza), picco RAM host
  - metriche di compute: GFLOPs forward (torch flop counter, best effort)

Output (semi-lavorati per grafici/tabelle):
  <out_dir>/runs_raw.csv    -> una riga per (combo, seed), scritta in append run-by-run
  <out_dir>/epochs.csv      -> formato long: una riga per (run, epoca)
  <out_dir>/runs_agg.csv    -> aggregato mean/std sui seed, una riga per combo
  <out_dir>/search_space.json, static_run_params.yaml

Uso:
    python -m src.utils.computational_cost --tag batchsize
    python -m src.utils.computational_cost --search-space my_space.json --resume
    python -m src.utils.computational_cost --no-flops --seeds 654 897

Note metodologiche:
  - variando batch_size il tempo/epoca non e' direttamente confrontabile (cambia il
    numero di batch): usare time_per_batch_train_ms / train_throughput_samples_s.
  - i GFLOPs sono forward-only su un batch reale; il costo di un training step e'
    approssimabile a ~3x il forward (fwd + bwd), riportato come train_gflops_per_batch_est.
"""

import os

os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0+PTX"  # per nuove GPU

import argparse
import csv
import gc
import hashlib
import itertools
import json
import platform
import socket
import statistics
import time
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from src.config import Parameters
from src.dataset.utils import setup_seed
from src.training.train import _pbar, validate, test
from src.training.training_module import Training
from src.utils.utils import get_datamodule

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:  # torch troppo vecchio
    FlopCounterMode = None

try:
    import psutil
except ImportError:
    psutil = None

NAN = float('nan')

# Colonne metriche del csv principale (header fisso -> append sicuro)
METRIC_COLUMNS = [
    # task
    'val_mse', 'val_rmse', 'val_mae',
    'test_mse', 'test_rmse', 'test_mae',
    # tempi
    'train_time_total_s', 'train_only_time_s', 'val_time_s', 'test_time_s',
    'epochs_run', 'early_stopped', 'best_epoch', 'time_to_best_s',
    'epoch_time_mean_s', 'epoch_time_std_s', 'epoch_time_median_s',
    'epoch_time_first_s', 'epoch_time_last_s',
    'time_per_batch_train_ms', 'train_throughput_samples_s',
    'test_time_per_batch_ms',
    'infer_latency_batch_ms', 'infer_latency_per_sample_ms', 'infer_batch_size_eff',
    'setup_time_s',
    # modello
    'n_params_total', 'n_params_trainable', 'model_size_mb', 'optimizer_state_mb',
    # memoria
    'vram_peak_train_mb', 'vram_reserved_peak_train_mb', 'vram_peak_infer_mb',
    'vram_model_mb', 'ram_peak_mb',
    # compute
    'fwd_gflops_per_batch', 'fwd_gflops_per_sample', 'train_gflops_per_batch_est',
    'total_train_gflops_est',
    # dati
    'input_dim', 'npatch', 'patch_layer',
    'n_train_batches', 'n_val_batches', 'n_test_batches', 'n_train_samples_epoch',
]

ENV_COLUMNS = ['gpu_name', 'device', 'torch_version', 'cuda_version', 'hostname', 'python_version']

EPOCH_COLUMNS = [
    'run_id', 'combo_id', 'seed', 'epoch',
    'train_time_s', 'val_time_s', 'epoch_time_s', 'cum_time_s',
    'train_loss', 'train_mse', 'train_rmse', 'train_mae', 'train_graph_loss', 'learning_rate',
    'val_loss', 'val_mse', 'val_rmse', 'val_mae', 'val_graph_loss',
    'vram_peak_epoch_mb', 'n_samples',
]


# --------------------------------------------------------------------------- #
# helper
# --------------------------------------------------------------------------- #
def build_combinations(search_space: dict) -> list:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def combo_id(combo: dict) -> str:
    """Hash stabile della combinazione, per join / resume."""
    payload = json.dumps(combo, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:10]


def _sync(device):
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize()


def _cuda_available(device) -> bool:
    return torch.cuda.is_available() and str(device).startswith('cuda')


def _reset_vram_peak(device):
    if _cuda_available(device):
        torch.cuda.reset_peak_memory_stats()


def _vram_peak_mb(device) -> float:
    if _cuda_available(device):
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return NAN


def _vram_reserved_peak_mb(device) -> float:
    if _cuda_available(device):
        return torch.cuda.max_memory_reserved() / 1024 ** 2
    return NAN


def _vram_allocated_mb(device) -> float:
    if _cuda_available(device):
        return torch.cuda.memory_allocated() / 1024 ** 2
    return NAN


def _ram_mb() -> float:
    if psutil is None:
        return NAN
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _model_stats(model) -> dict:
    params = list(model.parameters())
    n_total = sum(p.numel() for p in params)
    n_train = sum(p.numel() for p in params if p.requires_grad)
    size_b = sum(p.numel() * p.element_size() for p in params)
    size_b += sum(b.numel() * b.element_size() for b in model.buffers())
    return {'n_params_total': n_total,
            'n_params_trainable': n_train,
            'model_size_mb': size_b / 1024 ** 2}


def _optimizer_state_mb(optimizer) -> float:
    if optimizer is None:
        return NAN
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total / 1024 ** 2


def _to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _batch_n_samples(batch_dict) -> int:
    for key in ('observed_data', 'tp_to_predict', 'data_to_predict', 'x'):
        v = batch_dict.get(key)
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            return int(v.shape[0])
    for v in batch_dict.values():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            return int(v.shape[0])
    return 0


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


# --------------------------------------------------------------------------- #
# training instrumentato (copia di src.training.train.train + strumentazione)
# --------------------------------------------------------------------------- #
def train_instrumented(training, data_module, run_params) -> tuple:
    """Come train(), ma restituisce (summary: dict, history: list[dict])."""
    training.configure_optimizers()

    early_stop_counter = 0
    log_every = 50
    early_stop_best = float('inf')

    history = []
    train_times, val_times = [], []
    best_epoch, time_to_best = -1, NAN
    best_val_so_far = float('inf')
    n_samples_epoch = 0
    n_batches_done = 0
    ram_peak = _ram_mb()
    vram_peak_train = 0.0
    vram_reserved_peak = 0.0
    early_stopped = False

    _reset_vram_peak(run_params.device)
    t_wall_start = time.perf_counter()

    for epoch in range(run_params.max_epochs):
        training.model.train()
        train_metrics_accum = defaultdict(list)
        epoch_samples = 0

        _reset_vram_peak(run_params.device)
        _sync(run_params.device)
        t_train = time.perf_counter()

        n_train = data_module["n_train_batches"]
        pbar = _pbar(
            itertools.islice(data_module["train_dataloader"], n_train),
            desc=f"Epoch {epoch + 1}/{run_params.max_epochs} [train]",
            enable=run_params.enable_progress_bar,
            total=n_train,
            mininterval=1.0,
            miniters=50,
        )
        for step, batch in enumerate(pbar):
            if batch is not None:
                batch_dict = _to_device(batch, run_params.device)
                epoch_samples += _batch_n_samples(batch_dict)
                n_batches_done += 1

                metrics = training.training_step(batch_dict)
                for k, v in metrics.items():
                    train_metrics_accum[k].append(v)

                if (step + 1) % log_every == 0:
                    pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in train_metrics_accum.items()})

        _sync(run_params.device)
        train_time = time.perf_counter() - t_train
        train_times.append(train_time)
        n_samples_epoch = max(n_samples_epoch, epoch_samples)

        avg_train = {k: _safe_float(np.mean(v)) for k, v in train_metrics_accum.items()}

        if run_params.enable_progress_bar:
            print(f"Epoch {epoch + 1}/{run_params.max_epochs} | " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in avg_train.items()))
        print(f'Epoch time: {train_time:.3f}s')

        avg_val, val_time = {}, 0.0
        if (epoch + 1) % run_params.check_val_every_n_epoch == 0:
            _sync(run_params.device)
            t_val = time.perf_counter()
            avg_val = validate(training, data_module, run_params)
            _sync(run_params.device)
            val_time = time.perf_counter() - t_val
            val_times.append(val_time)

            training.on_validation_epoch_end(avg_val)
            training.scheduler.step(avg_val['val_mse'])

            if run_params.enable_progress_bar:
                print("  Val | " + " | ".join(f"{k}: {v:.4f}" for k, v in avg_val.items()))

            if avg_val['val_mse'] < best_val_so_far:
                best_val_so_far = avg_val['val_mse']
                best_epoch = epoch + 1
                time_to_best = time.perf_counter() - t_wall_start

            if run_params.early_stop_callback_flag:
                if avg_val['val_mse'] < early_stop_best:
                    early_stop_best = avg_val['val_mse']
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= run_params.early_stop_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        early_stopped = True

        epoch_vram = _vram_peak_mb(run_params.device)
        epoch_vram_res = _vram_reserved_peak_mb(run_params.device)
        if epoch_vram == epoch_vram:  # non NaN
            vram_peak_train = max(vram_peak_train, epoch_vram)
        if epoch_vram_res == epoch_vram_res:
            vram_reserved_peak = max(vram_reserved_peak, epoch_vram_res)
        ram_peak = max(ram_peak, _ram_mb())

        row = {'epoch': epoch + 1,
               'train_time_s': train_time,
               'val_time_s': val_time,
               'epoch_time_s': train_time + val_time,
               'cum_time_s': time.perf_counter() - t_wall_start,
               'vram_peak_epoch_mb': epoch_vram,
               'n_samples': epoch_samples}
        row.update({k: _safe_float(v) for k, v in avg_train.items() if k in EPOCH_COLUMNS})
        row.update({k: _safe_float(v) for k, v in avg_val.items() if k in EPOCH_COLUMNS})
        history.append(row)

        if early_stopped:
            break

    total_time = time.perf_counter() - t_wall_start
    epoch_times = [r['epoch_time_s'] for r in history]
    train_only = float(np.sum(train_times)) if train_times else NAN
    total_samples = sum(r['n_samples'] for r in history)

    summary = {
        'train_time_total_s': total_time,
        'train_only_time_s': train_only,
        'val_time_s': float(np.sum(val_times)) if val_times else 0.0,
        'epochs_run': len(history),
        'early_stopped': bool(early_stopped),
        'best_epoch': best_epoch,
        'time_to_best_s': time_to_best,
        'epoch_time_mean_s': float(np.mean(epoch_times)) if epoch_times else NAN,
        'epoch_time_std_s': float(statistics.stdev(epoch_times)) if len(epoch_times) > 1 else 0.0,
        'epoch_time_median_s': float(np.median(epoch_times)) if epoch_times else NAN,
        'epoch_time_first_s': epoch_times[0] if epoch_times else NAN,
        'epoch_time_last_s': epoch_times[-1] if epoch_times else NAN,
        'time_per_batch_train_ms': (train_only / n_batches_done * 1000) if n_batches_done else NAN,
        'train_throughput_samples_s': (total_samples / train_only) if train_only else NAN,
        'n_train_samples_epoch': n_samples_epoch,
        'vram_peak_train_mb': vram_peak_train if _cuda_available(run_params.device) else NAN,
        'vram_reserved_peak_train_mb': vram_reserved_peak if _cuda_available(run_params.device) else NAN,
        'ram_peak_mb': ram_peak,
        '_total_samples_seen': total_samples,
    }
    return summary, history


# --------------------------------------------------------------------------- #
# misure di inferenza / flops
# --------------------------------------------------------------------------- #
def measure_inference(training, data_module, run_params, n_warmup=2, n_iter=10) -> dict:
    """Latenza di forward in eval su batch reali del test loader."""
    training.model.eval()
    loader = data_module["test_dataloader"]
    times, sizes = [], []

    _reset_vram_peak(run_params.device)
    with torch.no_grad():
        for i in range(n_warmup + n_iter):
            batch = next(loader)
            if batch is None:
                continue
            batch_dict = _to_device(batch, run_params.device)
            _sync(run_params.device)
            t0 = time.perf_counter()
            training.forward(batch_dict)
            _sync(run_params.device)
            dt = time.perf_counter() - t0
            if i >= n_warmup:
                times.append(dt)
                sizes.append(_batch_n_samples(batch_dict))

    if not times:
        return {'infer_latency_batch_ms': NAN,
                'infer_latency_per_sample_ms': NAN,
                'infer_batch_size_eff': NAN,
                'vram_peak_infer_mb': _vram_peak_mb(run_params.device)}

    bs = float(np.mean(sizes)) if sizes else NAN
    lat = float(np.median(times)) * 1000
    return {'infer_latency_batch_ms': lat,
            'infer_latency_per_sample_ms': lat / bs if bs else NAN,
            'infer_batch_size_eff': bs,
            'vram_peak_infer_mb': _vram_peak_mb(run_params.device)}


def measure_flops(training, data_module, run_params) -> dict:
    """GFLOPs del forward su un batch reale. Best effort: NaN se non contabile."""
    out = {'fwd_gflops_per_batch': NAN,
           'fwd_gflops_per_sample': NAN,
           'train_gflops_per_batch_est': NAN}
    if FlopCounterMode is None:
        return out
    try:
        batch = next(data_module["test_dataloader"])
        if batch is None:
            return out
        batch_dict = _to_device(batch, run_params.device)
        n = _batch_n_samples(batch_dict)
        training.model.eval()
        counter = FlopCounterMode(display=False)
        # NB: niente no_grad, il ModuleTracker di FlopCounterMode registra hook sui
        # grad_fn e fallisce (AssertionError) su tensori senza grafo.
        with counter:
            fwd_out = training.forward(batch_dict)
        del fwd_out
        flops = counter.get_total_flops()
        gflops = flops / 1e9
        out['fwd_gflops_per_batch'] = gflops
        out['fwd_gflops_per_sample'] = gflops / n if n else NAN
        out['train_gflops_per_batch_est'] = 3.0 * gflops  # fwd + bwd
    except Exception as exc:  # layer custom non contabili, ecc.
        print(f'[flops] misura non riuscita: {type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        gc.collect()
        if _cuda_available(run_params.device):
            torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# esecuzione di un singolo run (combo, seed)
# --------------------------------------------------------------------------- #
def run_single(combo: dict, seed: int, global_config: dict, measure_flops_flag: bool) -> dict:
    run_params = Parameters().update(combo | global_config | {'seed': seed})
    if run_params.reproducible:
        setup_seed(seed)

    _reset_vram_peak(run_params.device)
    t_setup = time.perf_counter()

    data_module, run_params = get_datamodule(run_params)
    vram_before_model = _vram_allocated_mb(run_params.device)
    training = Training(run_params)
    _sync(run_params.device)
    setup_time = time.perf_counter() - t_setup

    res = {'status': 'ok', 'error': ''}
    res.update(_model_stats(training.model))
    res['vram_model_mb'] = _vram_allocated_mb(run_params.device) - vram_before_model
    res['setup_time_s'] = setup_time
    res.update({'input_dim': data_module.get('input_dim'),
                'npatch': getattr(run_params, 'npatch', None),
                'patch_layer': getattr(run_params, 'patch_layer', None),
                'n_train_batches': data_module.get('n_train_batches'),
                'n_val_batches': data_module.get('n_val_batches'),
                'n_test_batches': data_module.get('n_test_batches')})

    # Train
    summary, history = train_instrumented(training, data_module, run_params)
    total_samples = summary.pop('_total_samples_seen', 0)
    res.update(summary)
    res['optimizer_state_mb'] = _optimizer_state_mb(training.optimizer)
    res.update({'val_mse': training.best_mse,
                'val_rmse': training.best_rmse,
                'val_mae': training.best_mae})

    # Test
    _sync(run_params.device)
    t_test = time.perf_counter()
    res_test = test(training, data_module, run_params)
    _sync(run_params.device)
    test_time = time.perf_counter() - t_test
    res['test_time_s'] = test_time
    n_test_batches = data_module.get('n_test_batches') or 0
    res['test_time_per_batch_ms'] = (test_time / n_test_batches * 1000) if n_test_batches else NAN
    res.update({'test_mse': res_test[0].get('test_mse'),
                'test_rmse': res_test[0].get('test_rmse'),
                'test_mae': res_test[0].get('test_mae')})

    # Inferenza + flops
    res.update(measure_inference(training, data_module, run_params))
    if measure_flops_flag:
        flops = measure_flops(training, data_module, run_params)
        res.update(flops)
        gpb = flops.get('train_gflops_per_batch_est', NAN)
        n_batches = summary['epochs_run'] * (data_module.get('n_train_batches') or 0)
        res['total_train_gflops_est'] = gpb * n_batches if gpb == gpb else NAN
    else:
        res['total_train_gflops_est'] = NAN

    res['_history'] = history
    res['_run_params'] = run_params
    res['_total_samples'] = total_samples

    del training, data_module
    gc.collect()
    if _cuda_available(run_params.device):
        torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------- #
# I/O csv
# --------------------------------------------------------------------------- #
def env_info(device=None) -> dict:
    gpu = ''
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = 'unknown'
    return {'gpu_name': gpu,
            'device': str(device) if device is not None else '',
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda or '',
            'hostname': socket.gethostname(),
            'python_version': platform.python_version()}


def open_csv(path: str, header: list):
    exists = os.path.exists(path)
    f = open(path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore', restval='')
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


def load_done_runs(path: str) -> set:
    """(combo_id, seed) gia' completati con successo."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('status') == 'ok':
                done.add((row.get('combo_id'), str(row.get('seed'))))
    return done


def aggregate_runs(raw_csv: str, out_csv: str, param_keys: list) -> None:
    """Aggrega runs_raw.csv sui seed -> mean/std per combo."""
    try:
        import pandas as pd
    except ImportError:
        print('[agg] pandas non disponibile, salto runs_agg.csv')
        return

    df = pd.read_csv(raw_csv)
    if df.empty:
        return
    ok = df[df['status'] == 'ok'].copy()
    if ok.empty:
        print('[agg] nessun run valido')
        return

    metric_cols = [c for c in METRIC_COLUMNS if c in ok.columns]
    metric_cols = [c for c in metric_cols
                   if pd.api.types.is_numeric_dtype(pd.to_numeric(ok[c], errors='coerce'))]
    for c in metric_cols:
        ok[c] = pd.to_numeric(ok[c], errors='coerce')

    group_keys = ['combo_id'] + [k for k in param_keys if k in ok.columns]
    agg = ok.groupby(group_keys, dropna=False)[metric_cols].agg(['mean', 'std'])
    agg.columns = [f'{c}_{stat}' for c, stat in agg.columns]
    agg['n_seeds_ok'] = ok.groupby(group_keys, dropna=False).size()
    agg = agg.reset_index()

    n_tot = df.groupby(['combo_id']).size().rename('n_seeds_tot')
    agg = agg.merge(n_tot, on='combo_id', how='left')
    agg.to_csv(out_csv, index=False)
    print(f'Aggregato salvato in: {out_csv}')


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
DEFAULT_SEARCH_SPACE = {
    'dataset_name': ['activity', 'physionet', 'mimic', 'ushcn'],  # 'activity', 'physionet', 'mimic', 'ushcn'
    'batch_size': [64, 32, 16],  # 4, 8, 16,
    'gnn_name': ['GAT'],
    'hid_dim': [64],
    'gnn_layers': [2],
    'pool_num_heads': [16],
    'inner_mode': [1],
    'decoder_name': ['simple'],
    'dropout': [0.0],
    'lr': [1e-3],
}

DEFAULT_GLOBAL_CONFIG = {
    'save_ckpts': False,
    'early_stop_callback_flag': True,
    'save_logs': True,
    'reproducible': True,
}


def parse_args():
    p = argparse.ArgumentParser(description='Benchmark costo computazionale + metriche di task')
    p.add_argument('--search-space', type=str, default=None,
                   help='json con lo search space (default: DEFAULT_SEARCH_SPACE)')
    p.add_argument('--out-dir', type=str, default=None,
                   help='cartella di output (default: registry/experiments/computational_cost/<timestamp>_<tag>)')
    p.add_argument('--tag', type=str, default='cost', help='suffisso della cartella di output')
    p.add_argument('--seeds', type=int, nargs='+', default=[654, 897, 26])
    p.add_argument('--max-epochs', type=int, default=None, help='override di max_epochs')
    p.add_argument('--no-flops', dest='flops', action='store_false', help='disabilita la misura dei FLOPs')
    p.add_argument('--no-progress', dest='progress', action='store_false', help='disabilita le progress bar')
    p.add_argument('--resume', action='store_true', help='salta i run gia\' presenti in runs_raw.csv')
    p.set_defaults(flops=True, progress=True)
    return p.parse_args()


def main():
    args = parse_args()

    if args.search_space:
        with open(args.search_space, encoding='utf-8') as f:
            search_space = json.load(f)
    else:
        search_space = DEFAULT_SEARCH_SPACE

    global_config = dict(DEFAULT_GLOBAL_CONFIG)
    if args.max_epochs is not None:
        global_config['max_epochs'] = args.max_epochs
    global_config['enable_progress_bar'] = args.progress

    out_dir = args.out_dir or os.path.join(Parameters().registry_dir,
                                           'experiments',
                                           'computational_cost',
                                           f'{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_{args.tag}')
    os.makedirs(out_dir, exist_ok=True)
    global_config['logs_dir'] = out_dir

    combinations = build_combinations(search_space)
    param_keys = list(search_space.keys())
    print(f'Output dir: {out_dir}')
    print(f'Total combinations: {len(combinations)} x {len(args.seeds)} seeds = '
          f'{len(combinations) * len(args.seeds)} runs')

    # dump configurazione
    with open(os.path.join(out_dir, 'search_space.json'), 'w', encoding='utf-8') as f:
        json.dump({'search_space': search_space,
                   'global_config': {k: str(v) for k, v in global_config.items()},
                   'seeds': args.seeds,
                   'flops': args.flops}, f, indent=4, ensure_ascii=False)
    Parameters().update(combinations[0] | global_config).to_yaml(
        os.path.join(out_dir, 'static_run_params.yaml'))

    raw_path = os.path.join(out_dir, 'runs_raw.csv')
    epochs_path = os.path.join(out_dir, 'epochs.csv')
    done = load_done_runs(raw_path) if args.resume else set()
    if done:
        print(f'Resume: {len(done)} run gia\' completati, verranno saltati')

    header = (['run_id', 'combo_id', 'run_index', 'seed', 'timestamp', 'status', 'error']
              + param_keys + METRIC_COLUMNS + ENV_COLUMNS)
    raw_f, raw_writer = open_csv(raw_path, header)
    ep_f, ep_writer = open_csv(epochs_path, EPOCH_COLUMNS)

    try:
        for cont, combo in enumerate(combinations):
            cid = combo_id(combo)
            for seed in args.seeds:
                if (cid, str(seed)) in done:
                    print(f'[skip] combo {cont + 1}/{len(combinations)} seed {seed}')
                    continue

                run_id = f'{cid}_{seed}'
                print(f'\n=== Run {cont + 1}/{len(combinations)} | seed {seed} | {combo} ===')
                row = {'run_id': run_id,
                       'combo_id': cid,
                       'run_index': cont,
                       'seed': seed,
                       'timestamp': datetime.now().isoformat(timespec='seconds')}
                row.update(combo)

                t0 = time.perf_counter()
                try:
                    res = run_single(combo, seed, global_config, args.flops)
                    history = res.pop('_history', [])
                    run_params = res.pop('_run_params', None)
                    res.pop('_total_samples', None)
                    row.update(res)
                    row.update(env_info(getattr(run_params, 'device', None)))

                    for h in history:
                        h.update({'run_id': run_id, 'combo_id': cid, 'seed': seed})
                        ep_writer.writerow(h)
                    ep_f.flush()

                    print(f'[ok] {time.perf_counter() - t0:.1f}s | '
                          f'test_mse={row.get("test_mse")} | '
                          f'epochs={row.get("epochs_run")} | '
                          f'vram_train={row.get("vram_peak_train_mb")}')
                except torch.cuda.OutOfMemoryError as exc:
                    row.update({'status': 'oom', 'error': str(exc)[:300]})
                    row.update(env_info())
                    print(f'[OOM] combo {cont} seed {seed}: salto')
                except Exception as exc:
                    row.update({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'[:300]})
                    row.update(env_info())
                    print(f'[ERROR] combo {cont} seed {seed}: {type(exc).__name__}: {exc}')
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                raw_writer.writerow(row)
                raw_f.flush()
    finally:
        raw_f.close()
        ep_f.close()

    aggregate_runs(raw_path, os.path.join(out_dir, 'runs_agg.csv'), param_keys)
    print(f'\nFatto. Risultati in: {out_dir}')


if __name__ == '__main__':
    main()
