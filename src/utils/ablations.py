import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Known variable names per dataset
DATASET_VAR_NAMES = {
    "physionet": [
        'Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST',
        'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS',
        'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC',
    ],
    "ushcn": ["SNOW", "SNWD", "PRCP", "TMAX", "TMIN"],
    "activity": [
        f"{sensor}_{axis}"
        for sensor in ["ANKLE_L", "ANKLE_R", "CHEST", "BELT"]
        for axis in ["x", "y", "z"]
    ],
    # Ordered as they appear in full_dataset.csv; names from label_dict.csv
    "mimic": [
        'ALT', 'Alk.Phosphatase', 'Anion Gap', 'AST', 'Bicarbonate', 'Bilirubin',
        'Chloride', 'Creatinine', 'Glucose', 'Potassium', 'Sodium', 'Urea Nitrogen',
        'Basophils', 'Eosinophils', 'Hematocrit', 'Hemoglobin', 'Lymphocytes', 'MCH',
        'MCHC', 'MCV', 'Monocytes', 'Neutrophils', 'Platelets', 'PT', 'PTT', 'RDW',
        'RBC', 'WBC', 'pH', 'Sp.Gravity', 'Lactate', 'Calcium', 'Magnesium',
        'Phosphate', 'NaCl Flush', 'MgSO4', 'KCl', 'CaGluconate', 'Morphine',
        'CellSaver', 'Insulin-Reg', 'OR Crystal.', 'Solution', 'KCL Bolus', 'LR',
        'D5%', 'Piggyback', 'Nitroglycerin', 'pO2', 'Base Excess', 'TotCO2', 'pCO2',
        'Albumin', 'ChestTube1', 'Foley', 'Aspirin', 'KCl Drug', 'Docusate',
        'Phenylephrine', 'PRBC', 'MgSO4 Drug', 'Metoprolol', 'MgSO4 Bolus',
        'Metop.Tart', 'Bisacodyl', 'Midazolam', 'Gastric Meds', 'Furosemide',
        'Insulin-Glar', 'D5W Drug', 'K Phos', 'PO Intake', 'Sterile Water',
        'JackPratt1', 'GT Flush', 'Insulin-Hum', 'Pre-Admission', 'TF Residual',
        'Humulin-R', 'Pantoprazole', 'Lorazepam', 'Ultrafiltrate', 'Stool',
        'Hydralazine', 'OR EBL', 'ChestTube2', 'Heparin', 'Void',
        'Norepinephrine', 'D5 1/2NS', 'Albumin 5%', 'Ostomy', 'CondCath',
        'Gastric Tube', 'Fecal Bag', 'Urine Incont.',
    ],
}


def _compute_missing_rates(dataset):
    """
    Computes per-variable missing rate for each sample.

    Args:
        dataset: iterable of (record_id, tt, vals, mask) tuples.
                 mask shape: (T, D), 1=observed 0=missing.

    Returns:
        np.ndarray of shape (N, D) with missing rates in [0, 1].
    """
    rates = []
    for _, tt, vals, mask in dataset:
        m = mask.to('cpu').numpy() if hasattr(mask, 'numpy') else np.array(mask.to('cpu'))
        if m.ndim == 3:
            # PersonActivity: (T, n_sensors, 3) -> (T, D)
            m = m.reshape(m.shape[0], -1)
        T = m.shape[0]
        rates.append(1.0 - m.sum(axis=0) / T)
    return np.array(rates)  # (N, D)


def _plot_violin(rates, var_names, dataset_name, output_dir):
    N, D = rates.shape
    figsize = (max(10, D * 0.55), 6)
    fig, ax = plt.subplots(figsize=figsize)

    # sort variables by median missing rate (descending) for readability
    order = np.argsort(-np.median(rates, axis=0))
    sorted_rates = [rates[:, i] for i in order]
    sorted_names = [var_names[i] for i in order]

    parts = ax.violinplot(sorted_rates, positions=range(D), showmedians=True, showextrema=True)
    parts['cmedians'].set_color('red')

    ax.set_xticks(range(D))
    ax.set_xticklabels(sorted_names, rotation=60, ha='right', fontsize=max(5, 9 - D // 10))
    ax.set_ylabel("Missing rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Missing rate per variable  —  {dataset_name}  (N={N})")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    fname = os.path.join(output_dir, f"missing_rate_violin_{dataset_name}.pdf")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def _plot_heatmap(rates, var_names, dataset_name, output_dir, max_samples):
    N, D = rates.shape

    if N > max_samples:
        idx = np.sort(np.random.choice(N, max_samples, replace=False))
        rates = rates[idx]
        N = max_samples
        print('N > max_samples')

    # sort samples by average missing rate
    rates = rates[np.argsort(rates.mean(axis=1))]

    # sort variables by median missing rate (descending)
    var_order = np.argsort(-np.median(rates, axis=0))
    rates = rates[:, var_order]
    sorted_names = [var_names[i] for i in var_order]

    h = max(5, N * 0.015)
    w = max(8, D * 0.35)
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(rates, aspect='auto', interpolation='nearest', cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_xticks(range(D))
    # ax.set_xticklabels(sorted_names, rotation=60, ha='right', fontsize=max(5, 9 - D // 10))
    ax.set_yticks([])
    # ax.set_ylabel(f"Samples (N={N}, sorted by avg missing rate)")
    ax.set_ylabel(f"Samples")
    ax.set_xlabel(f"Channels")
    # ax.set_title(f"Missing rate heatmap  —  {dataset_name}")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Missing rate")

    fig.tight_layout()
    fname = os.path.join(output_dir, f"missing_rate_heatmap_{dataset_name}.pdf")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_missing_rate_per_variable(
    dataset,
    var_names=None,
    dataset_name="dataset",
    output_dir=".",
    plot_type="both",
    max_samples=500,
):
    """
    Plots the distribution of missing rate per variable across samples.

    Args:
        dataset:      iterable of (record_id, tt, vals, mask) tuples.
        var_names:    list of variable names. If None, tries DATASET_VAR_NAMES[dataset_name],
                      then falls back to generic "var_i" labels.
        dataset_name: string used in titles and output filenames.
        output_dir:   directory where plots are saved.
        plot_type:    "violin", "heatmap", or "both".
        max_samples:  max samples shown in the heatmap (random subsample if exceeded).
    """
    rates = _compute_missing_rates(dataset)
    N, D = rates.shape

    if var_names is None:
        var_names = DATASET_VAR_NAMES.get(dataset_name.lower(), [f"var_{i}" for i in range(D)])

    if len(var_names) != D:
        var_names = [f"var_{i}" for i in range(D)]

    os.makedirs(output_dir, exist_ok=True)

    if plot_type in ("violin", "both"):
        _plot_violin(rates, var_names, dataset_name, output_dir)
    if plot_type in ("heatmap", "both"):
        _plot_heatmap(rates, var_names, dataset_name, output_dir, max_samples)
    if plot_type not in ("violin", "heatmap", "both"):
        raise ValueError(f"plot_type must be 'violin', 'heatmap', or 'both', got '{plot_type}'")


if __name__ == "__main__":
    import torch
    from src.config import Parameters, initialize_configuration
    from src.dataset.physionet import PhysioNet
    from src.dataset.mimic import MIMIC
    from src.dataset.ushcn import USHCN
    from src.dataset.person_activity import PersonActivity

    args = initialize_configuration()
    device = torch.device("cpu")
    plot_type = "heatmap"  # ["violin", "heatmap", "both"]
    max_samples = 500
    datasets = ["physionet", "mimic", "ushcn", "activity"]
    output_dir = "plots/ablations_2"

    # physionet_dataset = PhysioNet(os.path.join(args.data_dir, args.dataset_name),
    #           quantization=args.quantization,
    #           download=False,
    #           n_samples=args.n,
    #           device=args.device)
    loaders = {
        "physionet": lambda: PhysioNet(os.path.join(args.data_dir, "physionet"),
              quantization=args.quantization,
              download=False,
              n_samples=args.n,
              device=args.device).data,
        "mimic": lambda: MIMIC(
            os.path.join(args.data_dir, "mimic"),
            device=device
        ).data,
        "ushcn": lambda: USHCN(
            os.path.join(args.data_dir, "ushcn"),
            device=device
        ).data,
        "activity": lambda: PersonActivity(
            os.path.join(args.data_dir, "activity"),
            device=device
        ).data,
    }

    for name in datasets:
        print(f"\n--- {name} ---")
        data = loaders[name]()
        plot_missing_rate_per_variable(
            dataset=data,
            dataset_name=name,
            output_dir=output_dir,
            plot_type=plot_type,
            max_samples=max_samples,
        )
