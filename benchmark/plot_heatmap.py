"""
Heatmap of model MAE across molecule subsets and charge categories.

Rows  : models
Cols  : Overall | Small Ligands | Large Ligands | Pentapeptides | Dimers | Neutral | Charged
Values: mean MAE (kcal mol⁻¹)
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem

KJ_TO_KCAL = 1 / 4.184

HDF5_PATH   = "/Users/franzgorlich/Documents/GitHub/cheminformer/benchmark/SPICE-test.hdf5"
MAE_DIR     = "/Users/franzgorlich/Documents/GitHub/cheminformer/aws_results/g7e.4xlarge/mae_per_mol"
OUTPUT_PATH = "/Users/franzgorlich/Documents/GitHub/cheminformer/benchmark/heatmap_g7e.png"

PRETTY_MODEL = {
    "AIMNet2":        "AIMNet2",
    "Egret-1":        "Egret-1",
    "FeNNix-Bio1_S_": "FeNNix-Bio1(S)",
    "MACE-MH-1":      "MACE-MH-1",
    "MACE-OFF23_L_":  "MACE-OFF23(L)",
    "MACE-OFF23_S_":  "MACE-OFF23(S)",
    "MACE-OFF24_M_":  "MACE-OFF24(M)",
    "MACE-OMOL-0":    "MACE-OMOL-0",
    "MACELES-OFF":    "MACELES-OFF",
    "Orb-v3-omol":    "Orb-v3-omol",
    "UMA-m-1":        "UMA-m-1",
    "UMA-s-1":        "UMA-s-1",
    "polar-1-l":      "Polar-1(L)",
    "polar-1-m":      "Polar-1(M)",
    "polar-1-s":      "Polar-1(S)",
}

# Column order and display labels
COL_KEYS   = ["Overall", "Small Ligands", "Large Ligands", "Pentapeptides", "Dimers", "Neutral", "Charged"]
COL_LABELS = ["Overall", "Small\nLigands", "Large\nLigands", "Penta-\npeptides", "Dimers", "Neutral", "Charged"]


def build_metadata(hdf5_path):
    rows = []
    with h5py.File(hdf5_path, "r") as f:
        for name in f.keys():
            n_atoms    = len(f[name]["atomic_numbers"][()])
            smiles_raw = f[name]["smiles"][()][0]
            smiles     = smiles_raw.decode() if isinstance(smiles_raw, bytes) else str(smiles_raw)
            mol        = Chem.MolFromSmiles(smiles)
            charge     = Chem.GetFormalCharge(mol) if mol else 0

            if " " in name:
                subset = "Dimers"
            elif "-" in name:
                subset = "Pentapeptides"
            elif n_atoms <= 50:
                subset = "Small Ligands"
            else:
                subset = "Large Ligands"

            rows.append({"name": name, "n_atoms": n_atoms, "charge": charge, "subset": subset})
    return pd.DataFrame(rows)


def load_model_results(mae_dir):
    model_dfs = {}
    for fname in sorted(os.listdir(mae_dir)):
        if not fname.endswith(".csv"):
            continue
        model_name = fname.replace(".csv", "")
        df = pd.read_csv(os.path.join(mae_dir, fname))
        model_dfs[model_name] = df.set_index("name")["mae_kj_mol"] * KJ_TO_KCAL
    return model_dfs


def build_matrix(meta, model_dfs):
    """Build (models × columns) matrix in kcal/mol."""
    model_names = list(model_dfs.keys())
    matrix = pd.DataFrame(index=model_names, columns=COL_KEYS, dtype=float)

    groups = {
        "Overall":       meta["name"].tolist(),
        "Small Ligands": meta.loc[meta["subset"] == "Small Ligands", "name"].tolist(),
        "Large Ligands": meta.loc[meta["subset"] == "Large Ligands", "name"].tolist(),
        "Pentapeptides": meta.loc[meta["subset"] == "Pentapeptides", "name"].tolist(),
        "Dimers":        meta.loc[meta["subset"] == "Dimers",        "name"].tolist(),
        "Neutral":       meta.loc[meta["charge"] == 0,               "name"].tolist(),
        "Charged":       meta.loc[meta["charge"] != 0,               "name"].tolist(),
    }

    for mname, series in model_dfs.items():
        for col, names in groups.items():
            vals = series.reindex(names).dropna()
            matrix.loc[mname, col] = vals.mean() if len(vals) else np.nan

    return matrix, {k: len(v) for k, v in groups.items()}


def main():
    print("Loading molecule metadata …")
    meta = build_metadata(HDF5_PATH)

    print("Loading model results …")
    model_dfs = load_model_results(MAE_DIR)

    matrix, counts = build_matrix(meta, model_dfs)

    # Sort models by overall MAE (best first)
    matrix = matrix.sort_values("Overall")
    row_labels = [PRETTY_MODEL.get(m, m) for m in matrix.index]

    data = matrix.values.astype(float)
    vmin = np.nanmin(data)
    vmax = np.nanpercentile(data[~np.isnan(data)], 95)

    # ── figure ────────────────────────────────────────────────────────────────
    n_rows, n_cols = data.shape
    fig_w = 2.0 + n_cols * 1.0
    fig_h = 1.0 + n_rows * 0.55
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # x-axis: column categories with counts
    x_labels = [f"{COL_LABELS[i]}\n(n={counts[COL_KEYS[i]]})" for i in range(len(COL_KEYS))]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # y-axis: model names
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # vertical separator after "Dimers" (index 4) and before "Neutral" (index 5)
    ax.axvline(x=4.5, color="white", linewidth=2.5)

    # cell annotations
    for r in range(n_rows):
        for c in range(n_cols):
            val = data[r, c]
            if np.isnan(val):
                continue
            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if norm_val > 0.60 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=text_color)

    # colour bar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("Mean MAE (kcal mol⁻¹)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        "MAE (kcal mol⁻¹) — SPICE test set, g7e.4xlarge\n"
        "Models sorted by overall MAE (best → worst)",
        fontsize=10, fontweight="bold", pad=40,
    )

    fig.text(
        0.5, -0.01,
        "Colour scale capped at 95th percentile. White line separates subsets from charge groups.",
        ha="center", fontsize=7.5, color="gray",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
