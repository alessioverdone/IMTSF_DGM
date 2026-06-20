"""
analyze_experiments.py
----------------------
Script per analizzare i risultati di esperimenti da CSV.

Funzionalità:
  1. Normalizza le colonne 'inner_mode' e 'decoder_name':
     - inner_mode vuoto → 1
     - decoder_name vuoto → 'simple'
  2. Permette di specificare N combinazioni di filtri (colonne+valori)
     e seleziona, per ciascuna, la riga con test_mse_mean minimo.
  3. Combina i risultati in una tabella finale con una colonna extra
     'source_run_index' che indica l'indice originale della riga scelta.

Utilizzo
--------
Modifica la sezione "CONFIGURAZIONE" in fondo al file, poi esegui:

    python analyze_experiments.py

oppure importa le funzioni e usale dal tuo codice.
"""
import os

import pandas as pd
from typing import Any


# ─────────────────────────────────────────────────────────────
# 1. CARICAMENTO E NORMALIZZAZIONE
# ─────────────────────────────────────────────────────────────

def load_and_normalize(csv_path: str) -> pd.DataFrame:
    """
    Carica il CSV e normalizza le colonne inner_mode e decoder_name:
      - inner_mode  NaN/vuoto  → 1
      - decoder_name NaN/vuoto → 'simple'

    Restituisce il DataFrame con una colonna 'source_run_index'
    che preserva l'indice originale della riga nel CSV.
    """
    print(f"\n📂 Caricamento: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalizza inner_mode
    df["inner_mode"] = (
        df["inner_mode"]
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""})
    )
    df.loc[df["inner_mode"] == "", "inner_mode"] = 1.0

    # Normalizza decoder_name
    df["decoder_name"] = (
        df["decoder_name"]
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""})
    )
    df.loc[df["decoder_name"] == "", "decoder_name"] = "simple"

    # Salva l'indice originale come colonna
    df = df.reset_index().rename(columns={"index": "source_run_index"})
    print(f"   {len(df)} righe caricate.\n")
    return df


# ─────────────────────────────────────────────────────────────
# 2. SELEZIONE DELLA RIGA MIGLIORE PER UNA COMBINAZIONE
# ─────────────────────────────────────────────────────────────

def select_best_row(
    df: pd.DataFrame,
    filters: dict[str, Any],
    metric: str = "test_mse_mean",
) -> pd.Series | None:
    """
    Filtra il DataFrame in base a 'filters' (dizionario colonna→valore)
    e restituisce la riga con il valore minimo nella colonna 'metric'.

    Parametri
    ---------
    df      : DataFrame normalizzato (output di load_and_normalize)
    filters : es. {"decoder_name": "crossAttn", "inner_mode": "2"}
    metric  : colonna su cui minimizzare (default: 'test_mse_mean')

    Restituisce None se nessuna riga corrisponde ai filtri.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in filters.items():
        mask &= df[col].astype(str) == str(val)

    subset = df[mask]
    if subset.empty:
        print(f"  ⚠️  Nessuna riga trovata per i filtri: {filters}")
        return None

    best_idx = subset[metric].idxmin()
    return subset.loc[best_idx]


# ─────────────────────────────────────────────────────────────
# 3. COSTRUZIONE DELLA TABELLA FINALE
# ─────────────────────────────────────────────────────────────

def build_results_table(
    df: pd.DataFrame,
    combinations: list[dict[str, Any]],
    metric: str = "test_mse_mean",
) -> pd.DataFrame:
    """
    Per ogni combinazione di filtri in 'combinations', seleziona la riga
    migliore e le assembla in un unico DataFrame.

    Parametri
    ---------
    df           : DataFrame normalizzato
    combinations : lista di dizionari filtro, es.
                   [
                     {"decoder_name": "simple",    "inner_mode": "1"},
                     {"decoder_name": "simple",    "inner_mode": "2"},
                     {"decoder_name": "crossAttn", "inner_mode": "2"},
                   ]
    metric       : metrica da minimizzare

    Restituisce un DataFrame con tante righe quante le combinazioni
    (le combinazioni senza match vengono saltate con un avviso).
    """
    rows = []
    for combo in combinations:
        print(f"→ Elaboro filtro: {combo}")
        best = select_best_row(df, combo, metric=metric)
        if best is not None:
            rows.append(best)

    if not rows:
        print("Nessun risultato trovato per nessuna combinazione.")
        return pd.DataFrame()

    result = pd.DataFrame(rows).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────
# 4. MAIN – CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = '/home/user/Scrivania/PhD/DGM/docs/logs esperimenti/raw_logs'
    csv_ = os.path.join(root, 'all_logs_v2.csv')
    metric = "test_mse_mean"  # Metrica da minimizzare

    COMBINATIONS = [
        {"dataset_name": "ushcn", "decoder_name": "simple", "inner_mode": 1.0},
        {"dataset_name": "ushcn", "decoder_name": "simple", "inner_mode": 2.0},
        {"dataset_name": "ushcn", "decoder_name": "simple", "inner_mode": 3.0},
        {"dataset_name": "ushcn", "decoder_name": "simple", "inner_mode": 4.0},
        {"dataset_name": "ushcn", "decoder_name": "crossAttn", "inner_mode": 1.0},
        {"dataset_name": "ushcn", "decoder_name": "crossAttn", "inner_mode": 2.0},
        {"dataset_name": "ushcn", "decoder_name": "crossAttn", "inner_mode": 3.0},
        {"dataset_name": "ushcn", "decoder_name": "crossAttn", "inner_mode": 4.0},
        {"dataset_name": "ushcn", "decoder_name": "filmSwiglu", "inner_mode": 1.0},
        {"dataset_name": "ushcn", "decoder_name": "filmSwiglu", "inner_mode": 2.0},
        {"dataset_name": "ushcn", "decoder_name": "filmSwiglu", "inner_mode": 3.0},
        {"dataset_name": "ushcn", "decoder_name": "filmSwiglu", "inner_mode": 4.0},
    ]
    OUTPUT_PATH = "best_results_ushcn.csv"
    df = load_and_normalize(csv_)


    print("🔍 Selezione migliori righe per ogni combinazione:\n")
    results = build_results_table(df, COMBINATIONS, metric=metric)

    if not results.empty:
        print(f"\n✅ Tabella finale ({len(results)} righe):\n")
        # Run,dataset_name,dropout,lr,batch_size,gnn_name,hid_dim,gnn_layers,pool_num_heads,val_mse_mean,val_mse_std,
        # val_rmse_mean,val_rmse_std,val_mae_mean,val_mae_std,test_mse_mean,test_mse_std,test_rmse_mean,test_rmse_std,
        # test_mae_mean,test_mae_std,inner_mode,decoder_name,max_epochs,source_folder
        # Mostra solo le colonne più rilevanti nel terminale
        display_cols = [
            "source_run_index", "decoder_name", "inner_mode",
            "test_mse_mean", "test_mse_std",
            "test_mae_mean", "test_mae_std",
        ]
        # Filtra solo le colonne esistenti
        display_cols = [c for c in display_cols if c in results.columns]
        print(results[display_cols].to_string(index=False))

        # results.to_csv(os.path.join(root, OUTPUT_PATH), index=False)
        print(f"\n💾 Salvato in: {OUTPUT_PATH}")
    else:
        print("Nessun risultato da salvare.")