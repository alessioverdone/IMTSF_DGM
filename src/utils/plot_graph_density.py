import os
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from src.utils.ablations import DATASET_VAR_NAMES


# ===========================================================================
# STILI CONFIGURABILI  —  modifica qui per cambiare l'aspetto del grafo
# ===========================================================================

@dataclass
class NodeStyle:
    """
    Stile dei nodi (le osservazioni).

    Due modalità di disegno:
      - mode='marker' : usa scatter con un marker (dimensione in punti^2, fissa
                        rispetto allo zoom). Veloce. Per "fondere" i nodi vicini
                        aumenta `size` e tieni edgecolor='none'.
      - mode='bar'    : disegna ogni nodo come un rettangolo in UNITÀ DATI.
                        Con bar_width=1.0 i nodi su time-index consecutivi si
                        TOCCANO esattamente e formano una barra continua,
                        indipendentemente dallo zoom/figsize. Più lento ma è
                        il modo robusto per ottenere l'effetto "cosa unica".
    """
    mode: str = 'marker'        # 'marker' oppure 'bar'

    # --- parametri modalità 'marker' ---
    marker: str = 's'           # 's'=quadrato, 'o'=cerchio, 'D'=rombo, '^'=triangolo...
    size: float = 28.0          # area del marker in punti^2 (più grande => più "fusi")

    # --- parametri modalità 'bar' ---
    bar_width: float = 0.9      # larghezza in unità di time-index (1.0 = si toccano)
    bar_height: float = 0.45    # altezza in unità di riga (variabile)

    # --- comuni a entrambe le modalità ---
    edgecolor: str = 'none'     # bordo del nodo ('none' aiuta la fusione, 'white' separa)
    linewidth: float = 0.3      # spessore bordo del nodo
    alpha: float = 1.0
    per_variable_color: bool = True   # True: un colore per variabile (cmap)
    fixed_color: str = 'steelblue'    # usato solo se per_variable_color=False
    cmap_name: str = 'tab20'          # colormap usata quando per_variable_color=True
    zorder: int = 4                   # i nodi stanno SOTTO gli archi (vedi EdgeStyle.zorder)


@dataclass
class EdgeStyle:
    """
    Stile di un arco. Vale sia per intra- che per inter-canale.

    arrowstyle (forma dell'arco / punta):
        '-'      linea semplice, NESSUNA freccia
        '->'     freccia sottile a destinazione
        '<-'     freccia sottile all'origine
        '<->'    doppia freccia sottile
        '-|>'    freccia piena (triangolo) a destinazione
        '<|-'    freccia piena all'origine
        '<|-|>'  doppia freccia piena
        'simple' / 'fancy' / 'wedge'   frecce "spesse" stile patch

    linestyle:
        'solid', 'dashed', 'dotted', 'dashdot'
        oppure tupla custom, es. (0, (3, 2)) per un tratteggio su misura

    rad: curvatura dell'arco. 0.0 = retto; valori tipo 0.2 / -0.3 curvano l'arco.
    """
    draw: bool = True            # se False questo tipo di arco non viene disegnato
    arrowstyle: str = '->'
    color: str = 'gray'
    linestyle: str = 'solid'
    linewidth: float = 0.8
    alpha: float = 1.0
    mutation_scale: float = 8.0  # dimensione della punta della freccia
    rad: float = 0.0             # curvatura (0 = linea retta; con nodi grandi alza es. 0.3)
    label: str = None            # etichetta in legenda (None => default)
    zorder: int = 2              # gli archi stanno SOPRA i nodi così restano visibili


# ---- Default usati se non passi nulla (replicano il comportamento originale) ----
DEFAULT_NODE_STYLE    = NodeStyle()
DEFAULT_INTRA_STYLE   = EdgeStyle(arrowstyle='->', color='gray',
                                  linestyle='solid', linewidth=0.8,
                                  label='intra-channel')
DEFAULT_INTER_STYLE_A = EdgeStyle(arrowstyle='->', color='royalblue',
                                  linestyle='dashed', linewidth=0.8,
                                  label='inter-channel')          # usato in mode 1 e 3
DEFAULT_INTER_STYLE_B = EdgeStyle(arrowstyle='->', color='tomato',
                                  linestyle='dashed', linewidth=0.8,
                                  label='inter-channel (B)')      # usato in mode 2 e 3


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
    # intra: catena di osservazioni consecutive PER CANALE (anche con gap temporali).
    # Per ogni variabile con k osservazioni => k-1 archi (path graph sulle osservazioni).
    n_obs_per_var = mask_2d.sum(axis=0)                       # (D,)
    n_intra = int(np.maximum(n_obs_per_var - 1, 0).sum())

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
    fname = os.path.join(output_dir, f"graph_stats_{dataset_name}.pdf")
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
    fname = os.path.join(output_dir, f"graph_scatter_{dataset_name}.pdf")
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

    # fig.suptitle("Graph structure — dataset comparison", fontsize=12)
    fig.tight_layout()
    fname = os.path.join(output_dir, "graph_stats_comparison.pdf")
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


def _draw_nodes(ax, active_vars, mask_2d, y_pos, colors, node_style):
    """Disegna i nodi secondo node_style.mode ('marker' o 'bar')."""
    if node_style.mode == 'bar':
        for v in active_vars:
            obs_t = np.where(mask_2d[:, v])[0]
            for t in obs_t:
                rect = Rectangle(
                    (t - node_style.bar_width / 2.0,
                     y_pos[v] - node_style.bar_height / 2.0),
                    node_style.bar_width, node_style.bar_height,
                    facecolor=colors[v], edgecolor=node_style.edgecolor,
                    linewidth=node_style.linewidth, alpha=node_style.alpha,
                    zorder=node_style.zorder,
                )
                ax.add_patch(rect)
    else:  # 'marker'
        for v in active_vars:
            obs_t = np.where(mask_2d[:, v])[0]
            ax.scatter(obs_t, [y_pos[v]] * len(obs_t),
                       color=colors[v],
                       marker=node_style.marker,
                       s=node_style.size,
                       edgecolors=node_style.edgecolor,
                       linewidths=node_style.linewidth,
                       alpha=node_style.alpha,
                       zorder=node_style.zorder)


def _draw_edge(ax, x0, y0, x1, y1, style):
    """Disegna un singolo arco (x0,y0)->(x1,y1) secondo lo EdgeStyle dato."""
    if not style.draw:
        return
    arrowprops = dict(
        arrowstyle=style.arrowstyle,
        color=style.color,
        linestyle=style.linestyle,
        lw=style.linewidth,
        alpha=style.alpha,
        mutation_scale=style.mutation_scale,
        connectionstyle=f'arc3,rad={style.rad}',
    )
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=arrowprops, zorder=style.zorder)


def _visualize_one_sample(tt_np, mask_2d, var_names, inter_mode, ax, title,
                          node_style=None, intra_style=None,
                          inter_style_a=None, inter_style_b=None):
    """
    Draws the GRAPE graph for one sample onto ax.
    - Y axis: active variable index (only variables with >=1 observation)
    - X axis: observation time index
    - Nodi: secondo `node_style` (quadrati/barre/marker, dimensione, colore)
    - Archi intra-canale: secondo `intra_style`
    - Archi inter-canale: secondo `inter_style_a` (mode 1/3) e `inter_style_b` (mode 2/3)
    """
    node_style    = node_style    or DEFAULT_NODE_STYLE
    intra_style   = intra_style   or DEFAULT_INTRA_STYLE
    inter_style_a = inter_style_a or DEFAULT_INTER_STYLE_A
    inter_style_b = inter_style_b or DEFAULT_INTER_STYLE_B

    T, D = mask_2d.shape

    # Keep only variables with at least one observation
    active_vars = [v for v in range(D) if mask_2d[:, v].any()]
    if not active_vars:
        ax.set_title(f"{title}\n(no observations)")
        return

    # Re-map to compact y positions
    y_pos = {v: i for i, v in enumerate(active_vars)}
    n_active = len(active_vars)

    # Colori dei nodi
    if node_style.per_variable_color:
        cmap = plt.cm.get_cmap(node_style.cmap_name, n_active)
        colors = {v: cmap(i) for i, v in enumerate(active_vars)}
    else:
        colors = {v: node_style.fixed_color for v in active_vars}

    # --- Nodi ---
    _draw_nodes(ax, active_vars, mask_2d, y_pos, colors, node_style)

    # --- Archi intra-canale: catena sulle osservazioni ordinate di ogni canale ---
    # (collega ogni osservazione alla successiva dello stesso canale, anche se
    #  separate da un gap temporale; estremi 1 vicino, interni 2 vicini)
    for v in active_vars:
        obs_t = np.where(mask_2d[:, v])[0]          # già ordinati in modo crescente
        for i in range(len(obs_t) - 1):
            t0, t1 = obs_t[i], obs_t[i + 1]
            _draw_edge(ax, t0, y_pos[v], t1, y_pos[v], intra_style)

    # --- Archi inter-canale ---
    for i, v in enumerate(active_vars[:-1]):
        v_next = active_vars[i + 1]
        obs_v = np.where(mask_2d[:, v])[0]
        obs_v1 = np.where(mask_2d[:, v_next])[0]
        if len(obs_v) == 0 or len(obs_v1) == 0:
            continue

        if inter_mode in (1, 3):
            src_t, dst_t = obs_v[-1], obs_v1[0]
            _draw_edge(ax, src_t, y_pos[v], dst_t, y_pos[v_next], inter_style_a)
        if inter_mode in (2, 3):
            src_t, dst_t = obs_v[0], obs_v1[0]
            _draw_edge(ax, src_t, y_pos[v], dst_t, y_pos[v_next], inter_style_b)

    ax.set_yticks(range(n_active))
    # ax.set_yticklabels([var_names[v] for v in active_vars], fontsize=max(5, 8 - n_active // 8))
    ax.set_xlabel("Time")
    ax.set_ylabel("Channels")

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, n_active - 0.5)          # utile soprattutto in mode='bar'
    # ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)

    # Legenda — Line2D riflette colore + linestyle reali di ciascun arco
    handles = []
    if intra_style.draw:
        handles.append(Line2D([0], [0], color=intra_style.color,
                              linestyle=intra_style.linestyle,
                              lw=max(1.2, intra_style.linewidth * 1.5),
                              label=intra_style.label or 'intra-channel'))
    if inter_style_a.draw and inter_mode in (1, 3):
        handles.append(Line2D([0], [0], color=inter_style_a.color,
                              linestyle=inter_style_a.linestyle,
                              lw=max(1.2, inter_style_a.linewidth * 1.5),
                              label=inter_style_a.label or 'inter-channel'))
    if inter_style_b.draw and inter_mode in (2, 3):
        handles.append(Line2D([0], [0], color=inter_style_b.color,
                              linestyle=inter_style_b.linestyle,
                              lw=max(1.2, inter_style_b.linewidth * 1.5),
                              label=inter_style_b.label or 'inter-channel (B)'))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc='upper right')


def visualize_graph_samples(dataset, stats, var_names, dataset_name,
                            output_dir, inter_mode=1, n_samples=3,
                            node_style=None, intra_style=None,
                            inter_style_a=None, inter_style_b=None):
    """
    Picks n_samples representative samples and draws the GRAPE graph for each.

    Gli stili (node_style / intra_style / inter_style_a / inter_style_b) sono
    opzionali: se None vengono usati i default a livello di modulo.
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
            node_style=node_style, intra_style=intra_style,
            inter_style_a=inter_style_a, inter_style_b=inter_style_b,
        )

    # fig.suptitle(f"GRAPE graph structure  —  {dataset_name}  (inter_mode={inter_mode})",
    #              fontsize=11)
    fig.tight_layout()
    fname = os.path.join(output_dir, f"graph_viz_{dataset_name}.pdf")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_graph_density(dataset, var_names=None, dataset_name="dataset",
                       output_dir=".", inter_mode=1, n_viz_samples=3,
                       node_style=None, intra_style=None,
                       inter_style_a=None, inter_style_b=None):
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
        node_style:     NodeStyle (opzionale) — aspetto dei nodi.
        intra_style:    EdgeStyle (opzionale) — archi intra-canale.
        inter_style_a:  EdgeStyle (opzionale) — archi inter-canale (mode 1/3).
        inter_style_b:  EdgeStyle (opzionale) — archi inter-canale (mode 2/3).

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
                            output_dir, inter_mode=inter_mode, n_samples=n_viz_samples,
                            node_style=node_style, intra_style=intra_style,
                            inter_style_a=inter_style_a, inter_style_b=inter_style_b)
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
    output_dir = "plots/ablations_3"
    datasets = ["physionet", "mimic", "ushcn", "activity"]

    # --- ESEMPIO di personalizzazione stili (rimuovi/modifica a piacere) ---
    my_node_style = NodeStyle(
        mode='marker',        # 'bar' per fusione perfetta dei nodi vicini
        marker='s',           # quadratini
        size=14,              # alza per farli "fondere"
        edgecolor='none',
    )
    my_intra_style = EdgeStyle(
        arrowstyle='-',      # freccia (usa '-' per linea liscia)
        color='#5277FE',
        linestyle='solid',
        linewidth=1.2,
        rad=0.0,              # con nodi grandi/fusi alza a ~0.3 per non farli coprire
    )
    my_inter_style_a = EdgeStyle(
        arrowstyle='-',     # freccia piena
        color='#9E3351',
        linestyle='dashed',
        linewidth=0.9,
        rad=0.0,
    )

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
            n_viz_samples=3,
            node_style=my_node_style,
            intra_style=my_intra_style,
            inter_style_a=my_inter_style_a,
        )

    # combined comparison across all datasets
    plot_graph_density_comparison(all_stats, output_dir)