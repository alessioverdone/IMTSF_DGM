import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.ablations import DATASET_VAR_NAMES


# ---------------------------------------------------------------------------
# Edge counting (replicates build_graph_from_mask_v2 analytically)
# ---------------------------------------------------------------------------

def _count_edges(mask_2d, inter_mode=1):
    """
    Counts intra- and inter-channel edges from a single-sample mask.

    Args:
        mask_2d:    np.ndarray (T, D), 1=observed 0=missing.
        inter_mode: 1, 2, 3 or 4 (same semantics as build_graph_from_mask_v2).

    Returns:
        n_intra, n_inter
    """
    # intra: consecutive time pairs (t, t+1) where mask[t,v]==1 AND mask[t+1,v]==1
    n_intra = int((mask_2d[:-1] & mask_2d[1:]).sum())

    has_obs = mask_2d.any(axis=0)          # (D,) bool
    n_nonempty = int(has_obs.sum())
    active_pairs = int((has_obs[:-1] & has_obs[1:]).sum())   # consecutive-var pairs

    if inter_mode in (1, 2):
        n_inter = active_pairs
    elif inter_mode == 3:
        n_inter = 2 * active_pairs
    else:                                   # mode 4
        n_inter = max(0, n_nonempty - 1) + max(0, n_nonempty // 2)

    return n_intra, n_inter


def _compute_graph_stats(dataset, inter_mode=1):
    """
    Computes per-sample graph statistics for an entire dataset.

    Returns dict with np.ndarrays:
        n_nodes, n_intra, n_inter, n_edges, density, missing_rate, intra_ratio
    """
    stats = defaultdict(list)

    for _, tt, vals, mask in dataset:
        m = mask.to('cpu').numpy() if hasattr(mask, 'numpy') else np.array(mask)
        if m.ndim == 3:
            m = m.reshape(m.shape[0], -1)
        m = m.astype(bool)
        T, D = m.shape

        n_nodes = int(m.sum())
        n_intra, n_inter = _count_edges(m, inter_mode)
        n_edges = n_intra + n_inter

        density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
        missing_rate = 1.0 - n_nodes / (T * D)
        intra_ratio = n_intra / n_edges if n_edges > 0 else 0.0

        stats['n_nodes'].append(n_nodes)
        stats['n_intra'].append(n_intra)
        stats['n_inter'].append(n_inter)
        stats['n_edges'].append(n_edges)
        stats['density'].append(density)
        stats['missing_rate'].append(missing_rate)
        stats['intra_ratio'].append(intra_ratio)

    return {k: np.array(v) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Plot 1 — density statistics per dataset
# ---------------------------------------------------------------------------

def _plot_density_stats(stats, dataset_name, output_dir):
    """
    4-panel figure: distributions of n_nodes, n_intra, n_inter, density.
    """
    metrics = [
        ('n_nodes',      'Nodes per sample'),
        ('n_intra',      'Intra-channel edges'),
        ('n_inter',      'Inter-channel edges'),
        ('density',      'Graph density'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (key, label) in zip(axes, metrics):
        data = stats[key]
        ax.violinplot([data], showmedians=True)
        ax.set_xticks([1])
        ax.set_xticklabels([dataset_name])
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.3)
        med = np.median(data)
        ax.text(1, med, f' {med:.3g}', va='bottom', fontsize=8, color='red')

    fig.suptitle(f"Graph structure statistics  —  {dataset_name}  (N={len(stats['n_nodes'])})",
                 fontsize=11)
    fig.tight_layout()
    fname = os.path.join(output_dir, f"graph_stats_{dataset_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 2 — scatter: missing rate vs. density / intra_ratio
# ---------------------------------------------------------------------------

def _plot_scatter(stats, dataset_name, output_dir):
    """
    Scatter plots: missing rate vs. graph density and vs. intra-edge ratio.
    """
    mr = stats['missing_rate']
    density = stats['density']
    intra_ratio = stats['intra_ratio']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(mr, density, s=6, alpha=0.4, color='steelblue')
    axes[0].set_xlabel("Missing rate (per sample)")
    axes[0].set_ylabel("Graph density")
    axes[0].set_title(f"Missing rate vs. density  —  {dataset_name}")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(mr, intra_ratio, s=6, alpha=0.4, color='darkorange')
    axes[1].set_xlabel("Missing rate (per sample)")
    axes[1].set_ylabel("Intra-edge ratio  (n_intra / n_edges)")
    axes[1].set_title(f"Missing rate vs. intra-edge ratio  —  {dataset_name}")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fname = os.path.join(output_dir, f"graph_scatter_{dataset_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 3 — multi-dataset comparison
# ---------------------------------------------------------------------------

def plot_graph_density_comparison(stats_per_dataset, output_dir):
    """
    Side-by-side box plots comparing graph stats across multiple datasets.

    Args:
        stats_per_dataset: dict  {dataset_name: stats_dict}
    """
    os.makedirs(output_dir, exist_ok=True)
    names = list(stats_per_dataset.keys())
    metrics = [
        ('n_nodes',     'Nodes per sample'),
        ('n_intra',     'Intra-channel edges'),
        ('n_inter',     'Inter-channel edges'),
        ('density',     'Graph density'),
        ('intra_ratio', 'Intra-edge ratio'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    for ax, (key, label) in zip(axes, metrics):
        data = [stats_per_dataset[n][key] for n in names]
        parts = ax.violinplot(data, showmedians=True)
        parts['cmedians'].set_color('red')
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Graph structure — dataset comparison", fontsize=12)
    fig.tight_layout()
    fname = os.path.join(output_dir, "graph_stats_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 4 — graph visualization for representative samples
# ---------------------------------------------------------------------------

def _pick_representative_samples(dataset, stats, n_samples=3):
    """
    Picks n_samples indices spread across the n_nodes distribution
    (low / median / high density), avoiding degenerate cases.
    """
    n_nodes = stats['n_nodes']
    valid = np.where(n_nodes > 2)[0]
    if len(valid) == 0:
        return []
    percentiles = np.percentile(n_nodes[valid], np.linspace(20, 80, n_samples))
    indices = []
    for p in percentiles:
        idx = valid[np.argmin(np.abs(n_nodes[valid] - p))]
        if idx not in indices:
            indices.append(int(idx))
    return indices


def _visualize_one_sample(tt_np, mask_2d, var_names, inter_mode, ax, title):
    """
    Draws the GRAPE graph for one sample onto ax.
    - Y axis: active variable index (only variables with ≥1 observation)
    - X axis: observation time index
    - Nodes: dots colored by variable
    - Intra edges: thin gray arrows within the same row
    - Inter edges: colored dashed arrows between rows
    """
    T, D = mask_2d.shape

    # Keep only variables with at least one observation
    active_vars = [v for v in range(D) if mask_2d[:, v].any()]
    if not active_vars:
        ax.set_title(f"{title}\n(no observations)")
        return

    # Re-map to compact y positions
    y_pos = {v: i for i, v in enumerate(active_vars)}
    n_active = len(active_vars)

    cmap = plt.cm.get_cmap('tab20', n_active)
    colors = {v: cmap(i) for i, v in enumerate(active_vars)}

    # Draw nodes
    for v in active_vars:
        obs_t = np.where(mask_2d[:, v])[0]
        ax.scatter(obs_t, [y_pos[v]] * len(obs_t),
                   color=colors[v], s=30, zorder=3, linewidths=0)

    arrow_kw = dict(arrowstyle='->', mutation_scale=8, lw=0.8)

    # Intra-channel edges
    for v in active_vars:
        obs_t = np.where(mask_2d[:, v])[0]
        for i in range(len(obs_t) - 1):
            t0, t1 = obs_t[i], obs_t[i + 1]
            if t1 == t0 + 1:                    # only consecutive time indices
                ax.annotate('', xy=(t1, y_pos[v]), xytext=(t0, y_pos[v]),
                             arrowprops=dict(color='gray', **arrow_kw))

    # Inter-channel edges
    inter_color = 'royalblue'
    for i, v in enumerate(active_vars[:-1]):
        v_next = active_vars[i + 1]
        obs_v = np.where(mask_2d[:, v])[0]
        obs_v1 = np.where(mask_2d[:, v_next])[0]
        if len(obs_v) == 0 or len(obs_v1) == 0:
            continue

        if inter_mode in (1, 3):
            src_t, dst_t = obs_v[-1], obs_v1[0]
            ax.annotate('', xy=(dst_t, y_pos[v_next]), xytext=(src_t, y_pos[v]),
                         arrowprops=dict(color=inter_color, linestyle='dashed', **arrow_kw))
        if inter_mode in (2, 3):
            src_t, dst_t = obs_v[0], obs_v1[0]
            ax.annotate('', xy=(dst_t, y_pos[v_next]), xytext=(src_t, y_pos[v]),
                         arrowprops=dict(color='tomato', linestyle='dashed', **arrow_kw))

    ax.set_yticks(range(n_active))
    ax.set_yticklabels([var_names[v] for v in active_vars], fontsize=max(5, 8 - n_active // 8))
    ax.set_xlabel("Time index")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)

    # Legend
    intra_patch = mpatches.Patch(color='gray', label='intra-channel')
    inter_patch = mpatches.Patch(color=inter_color, label='inter-channel')
    ax.legend(handles=[intra_patch, inter_patch], fontsize=7, loc='upper right')


def visualize_graph_samples(dataset, stats, var_names, dataset_name,
                             output_dir, inter_mode=1, n_samples=3):
    """
    Picks n_samples representative samples and draws the GRAPE graph for each.
    """
    indices = _pick_representative_samples(dataset, stats, n_samples)
    if not indices:
        print(f"No valid samples to visualize for {dataset_name}.")
        return

    samples = list(dataset)
    fig, axes = plt.subplots(1, len(indices), figsize=(7 * len(indices), max(4, len(var_names) * 0.18)))

    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        _, tt, vals, mask = samples[idx]
        m = mask.to('cpu').numpy() if hasattr(mask, 'numpy') else np.array(mask)
        if m.ndim == 3:
            m = m.reshape(m.shape[0], -1)
        m = m.astype(bool)

        n_nodes = stats['n_nodes'][idx]
        n_intra = stats['n_intra'][idx]
        n_inter = stats['n_inter'][idx]
        mr = stats['missing_rate'][idx]

        title = (f"sample #{idx}  |  nodes={n_nodes}  "
                 f"intra={n_intra}  inter={n_inter}  "
                 f"miss={mr:.2f}")
        _visualize_one_sample(
            tt.to('cpu').numpy() if hasattr(tt, 'numpy') else np.array(tt),
            m, var_names, inter_mode, ax, title,
        )

    fig.suptitle(f"GRAPE graph structure  —  {dataset_name}  (inter_mode={inter_mode})",
                 fontsize=11)
    fig.tight_layout()
    fname = os.path.join(output_dir, f"graph_viz_{dataset_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_graph_density(dataset, var_names=None, dataset_name="dataset",
                       output_dir=".", inter_mode=1, n_viz_samples=3):
    """
    Runs the full graph density analysis for one dataset:
      1. Computes per-sample stats (nodes, intra/inter edges, density).
      2. Saves violin plots of stats distribution.
      3. Saves scatter: missing rate vs. density / intra-ratio.
      4. Saves graph visualization for representative samples.

    Args:
        dataset:        iterable of (record_id, tt, vals, mask) tuples.
        var_names:      list of variable names (optional).
        dataset_name:   string used in titles and filenames.
        output_dir:     where to save plots.
        inter_mode:     1, 2, 3, or 4 — edge construction mode.
        n_viz_samples:  number of samples to visualize as graphs.

    Returns:
        stats dict (so caller can pass to plot_graph_density_comparison).
    """
    os.makedirs(output_dir, exist_ok=True)

    # infer D from first sample to set var_names
    first = next(iter(dataset))
    m0 = first[3]
    m0 = m0.to('cpu').numpy() if hasattr(m0, 'numpy') else np.array(m0)
    if m0.ndim == 3:
        m0 = m0.reshape(m0.shape[0], -1)
    D = m0.shape[1]

    if var_names is None:
        var_names = DATASET_VAR_NAMES.get(dataset_name.lower(), [f"var_{i}" for i in range(D)])
    if len(var_names) != D:
        var_names = [f"var_{i}" for i in range(D)]

    print(f"  Computing graph stats...")
    stats = _compute_graph_stats(dataset, inter_mode=inter_mode)

    _plot_density_stats(stats, dataset_name, output_dir)
    _plot_scatter(stats, dataset_name, output_dir)
    visualize_graph_samples(dataset, stats, var_names, dataset_name,
                             output_dir, inter_mode=inter_mode, n_samples=n_viz_samples)
    return stats


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from src.config import initialize_configuration
    from src.dataset.physionet import PhysioNet
    from src.dataset.mimic import MIMIC
    from src.dataset.ushcn import USHCN
    from src.dataset.person_activity import PersonActivity

    args = initialize_configuration()
    device = torch.device("cpu")
    inter_mode = 1          # change to match your main experiments
    output_dir = "plots/ablations_2"
    datasets = ["physionet", "mimic", "ushcn", "activity"]

    loaders = {
        "physionet": lambda: PhysioNet(
            os.path.join(args.data_dir, "physionet"),
            quantization=args.quantization,
            download=False,
            n_samples=args.n,
            device=device,
        ).data,
        "mimic": lambda: MIMIC(
            os.path.join(args.data_dir, "mimic"),
            device=device,
        ).data,
        "ushcn": lambda: USHCN(
            os.path.join(args.data_dir, "ushcn"),
            device=device,
        ).data,
        "activity": lambda: PersonActivity(
            os.path.join(args.data_dir, "activity"),
            device=device,
        ).data,
    }

    all_stats = {}
    for name in datasets:
        print(f"\n--- {name} ---")
        data = loaders[name]()
        all_stats[name] = plot_graph_density(
            dataset=data,
            dataset_name=name,
            output_dir=output_dir,
            inter_mode=inter_mode,
            n_viz_samples=1,
        )

    # combined comparison across all datasets
    plot_graph_density_comparison(all_stats, output_dir)
