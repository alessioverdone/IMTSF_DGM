import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.ablations import DATASET_VAR_NAMES


def _compute_gaps(dataset):
    """
    For each variable, collects all inter-observation gaps across all samples.
    Gaps are normalized by each sample's time span so they are in [0, 1] and
    comparable across datasets.

    Args:
        dataset: iterable of (record_id, tt, vals, mask) tuples.
                 tt shape: (T,), mask shape: (T, D) or (T, S, 3).

    Returns:
        gaps_by_var: list of D np.ndarrays, each containing all normalized gaps
                     for that variable (concatenated across samples).
        obs_by_var:  np.ndarray (D,) total observation counts per variable.
    """
    D = None
    gaps_by_var = None
    obs_by_var = None

    for _, tt, vals, mask in dataset:
        t = tt.to('cpu').numpy() if hasattr(tt, 'numpy') else np.array(tt)
        m = mask.to('cpu').numpy() if hasattr(mask, 'numpy') else np.array(mask)
        if m.ndim == 3:
            m = m.reshape(m.shape[0], -1)

        if D is None:
            D = m.shape[1]
            gaps_by_var = [[] for _ in range(D)]
            obs_by_var = np.zeros(D, dtype=np.int64)

        span = t[-1] - t[0] if len(t) > 1 else 1.0
        if span <= 0:
            span = 1.0

        for v in range(D):
            obs_times = t[m[:, v] == 1]
            obs_by_var[v] += len(obs_times)
            if len(obs_times) > 1:
                gaps_by_var[v].extend((np.diff(obs_times) / span).tolist())

    if D is None:
        return [], np.array([])

    return [np.array(g) for g in gaps_by_var], obs_by_var


def _plot_gap_distribution(gaps_by_var, dataset_name, output_dir):
    """
    Aggregate log-gap histogram + ECDF across all variables and samples.
    """
    all_gaps = np.concatenate([g for g in gaps_by_var if len(g) > 0])
    if len(all_gaps) == 0:
        print(f"No gaps found for {dataset_name}, skipping distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # log-gap histogram
    log_gaps = np.log1p(all_gaps)
    axes[0].hist(log_gaps, bins=60, density=True, color='steelblue', alpha=0.8, edgecolor='none')
    axes[0].set_xlabel("log(1 + gap / span)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Gap distribution  —  {dataset_name}  (N gaps={len(all_gaps):,})")
    axes[0].grid(alpha=0.3)

    # ECDF on log x-axis
    sorted_gaps = np.sort(all_gaps)
    ecdf = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)
    axes[1].plot(sorted_gaps, ecdf, linewidth=1.5, color='steelblue')
    axes[1].set_xlabel("Gap / span")
    axes[1].set_ylabel("ECDF")
    axes[1].set_xscale('log')
    axes[1].set_title(f"Gap ECDF (log scale)  —  {dataset_name}")
    axes[1].grid(alpha=0.3, which='both')

    for p, label in [(0.5, 'p50'), (0.9, 'p90'), (0.99, 'p99')]:
        val = np.quantile(all_gaps, p)
        axes[1].axvline(val, linestyle='--', linewidth=0.9, alpha=0.7, label=f"{label}={val:.3f}")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fname = os.path.join(output_dir, f"gap_distribution_{dataset_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def _plot_burstiness_per_variable(gaps_by_var, var_names, dataset_name, output_dir):
    """
    CV (std/mean) and median gap per variable, sorted by CV descending.
    Bar chart for D <= 25, heatmap otherwise.
    """
    D = len(gaps_by_var)

    cv = np.array([
        g.std() / g.mean() if len(g) > 1 and g.mean() > 0 else np.nan
        for g in gaps_by_var
    ])
    median_gap = np.array([
        np.median(g) if len(g) > 0 else np.nan
        for g in gaps_by_var
    ])

    valid = ~np.isnan(cv)
    order = np.concatenate([
        np.where(valid)[0][np.argsort(-cv[valid])],
        np.where(~valid)[0],
    ])
    cv_s = cv[order]
    med_s = median_gap[order]
    names_s = [var_names[i] for i in order]

    if D <= 25:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, D * 0.35)))
        y = range(D)

        bar_colors = ['#d62728' if (not np.isnan(v) and v > 1.0) else '#1f77b4' for v in cv_s]
        axes[0].barh(y, cv_s, color=bar_colors, alpha=0.8)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(names_s, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].axvline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='CV=1')
        axes[0].set_xlabel("CV of gaps (std / mean)")
        axes[0].set_title(f"Burstiness  —  {dataset_name}")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis='x', alpha=0.3)

        axes[1].barh(y, med_s, color='steelblue', alpha=0.8)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(names_s, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Median gap / span")
        axes[1].set_title(f"Median inter-obs gap  —  {dataset_name}")
        axes[1].grid(axis='x', alpha=0.3)

    else:
        h = max(6, D * 0.14)
        fig, axes = plt.subplots(1, 2, figsize=(10, h))
        fs = max(4, 8 - D // 20)

        for ax, raw, title, cmap in [
            (axes[0], cv_s,  f"Burstiness (CV)\n{dataset_name}", 'RdYlGn_r'),
            (axes[1], med_s, f"Median gap / span\n{dataset_name}", 'RdYlGn_r'),
        ]:
            im = ax.imshow(raw.reshape(-1, 1), aspect='auto', cmap=cmap,
                           vmin=np.nanmin(raw), vmax=np.nanmax(raw))
            ax.set_xticks([])
            ax.set_yticks(range(D))
            ax.set_yticklabels(names_s, fontsize=fs)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.tight_layout()
    fname = os.path.join(output_dir, f"gap_burstiness_{dataset_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_observation_gap_distribution(
    dataset,
    var_names=None,
    dataset_name="dataset",
    output_dir=".",
    plot_type="both",
):
    """
    Analyses the temporal distribution of observations via inter-observation gaps.

    Args:
        dataset:      iterable of (record_id, tt, vals, mask) tuples.
        var_names:    list of variable names. Falls back to DATASET_VAR_NAMES or "var_i".
        dataset_name: string used in titles and output filenames.
        output_dir:   directory where plots are saved.
        plot_type:    "distribution"  — aggregate log-gap histogram + ECDF
                      "burstiness"   — CV and median gap per variable
                      "both"         — both
    """
    if plot_type not in ("distribution", "burstiness", "both"):
        raise ValueError(f"plot_type must be 'distribution', 'burstiness', or 'both', got '{plot_type}'")

    gaps_by_var, obs_by_var = _compute_gaps(dataset)
    D = len(gaps_by_var)

    if var_names is None:
        var_names = DATASET_VAR_NAMES.get(dataset_name.lower(), [f"var_{i}" for i in range(D)])
    if len(var_names) != D:
        var_names = [f"var_{i}" for i in range(D)]

    os.makedirs(output_dir, exist_ok=True)

    if plot_type in ("distribution", "both"):
        _plot_gap_distribution(gaps_by_var, dataset_name, output_dir)
    if plot_type in ("burstiness", "both"):
        _plot_burstiness_per_variable(gaps_by_var, var_names, dataset_name, output_dir)


if __name__ == "__main__":
    import torch
    from src.config import initialize_configuration
    from src.dataset.physionet import PhysioNet
    from src.dataset.mimic import MIMIC
    from src.dataset.ushcn import USHCN
    from src.dataset.person_activity import PersonActivity

    args = initialize_configuration()
    device = torch.device("cpu")
    plot_type = "both"  # ["distribution", "burstiness", "both"]
    datasets = ["physionet", "mimic", "ushcn", "activity"]
    output_dir = "plots/ablations"

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

    for name in datasets:
        print(f"\n--- {name} ---")
        data = loaders[name]()
        plot_observation_gap_distribution(
            dataset=data,
            dataset_name=name,
            output_dir=output_dir,
            plot_type=plot_type,
        )
