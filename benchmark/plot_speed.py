"""
Speed heatmap: models (rows) × systems (cols), coloured by ms/step (log scale).
Non-ok cells are annotated with their failure reason and shown in grey.
"""

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

RESULTS_DIR = "/Users/franzgorlich/Documents/GitHub/cheminformer/aws_results/g5e.4xlarge"
OUTPUT_PATH = "/Users/franzgorlich/Documents/GitHub/cheminformer/benchmark/heatmap_speed_g5e.png"

# Systems to show, ordered by n_atoms
SYSTEM_ORDER = [
    "capped_ala", "chignolin", "ubiquitin",
    "2LZM", "1ZG4", "3N5G", "5G1P", "1B3B", "9VM6", "water_99k",
]
N_ATOMS = {
    "capped_ala": 22, "chignolin": 138, "ubiquitin": 602,
    "2LZM": 1427, "1ZG4": 2224, "3N5G": 2304,
    "5G1P": 13_193, "1B3B": 19_008, "9VM6": 53_700, "water_99k": 99_999,
}

PRETTY_MODEL = {
    "AIMNet2":        "AIMNet2",
    "AceFF-1.1":      "AceFF-1.1",
    "AceFF-2.0":      "AceFF-2.0",
    "Egret-1":        "Egret-1",
    "FeNNix-Bio1(M)": "FeNNix-Bio1(M)",
    "FeNNix-Bio1(S)": "FeNNix-Bio1(S)",
    "MACE-MH-1":      "MACE-MH-1",
    "MACE-OFF23(L)":  "MACE-OFF23(L)",
    "MACE-OFF23(S)":  "MACE-OFF23(S)",
    "MACE-OFF24(M)":  "MACE-OFF24(M)",
    "MACE-OMOL-0":    "MACE-OMOL-0",
    "MACELES-OFF":    "MACELES-OFF",
    "Orb-v3-omol":    "Orb-v3-omol",
    "UMA-m-1":        "UMA-m-1",
    "UMA-s-1":        "UMA-s-1",
    "polar-1-l":      "Polar-1(L)",
    "polar-1-m":      "Polar-1(M)",
    "polar-1-s":      "Polar-1(S)",
}


def failure_label(row):
    if row["status"] == "skipped":
        return "skip"
    if row["status"] == "killed":
        return "killed"
    if row["status"] == "timeout":
        return "t/o"
    if row["status"] == "error":
        err = str(row.get("error", ""))
        if "memory" in err.lower() or "OutOfMemory" in err:
            return "OOM"
        return "err"
    return ""


def load_data():
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(f"{RESULTS_DIR}/results_*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["system"].isin(SYSTEM_ORDER)]
    return df


def build_pivots(df):
    ok = df[df["status"] == "ok"].copy()

    ms_pivot = ok.pivot_table(index="model", columns="system", values="ms_per_step", aggfunc="mean")
    status_pivot = df.pivot_table(index="model", columns="system", values="status", aggfunc="first")
    label_pivot = df.apply(failure_label, axis=1)
    label_pivot = df.assign(fail_label=label_pivot).pivot_table(
        index="model", columns="system", values="fail_label", aggfunc="first"
    )

    all_models = sorted(df["model"].unique())
    ms_pivot    = ms_pivot.reindex(index=all_models, columns=SYSTEM_ORDER)
    status_pivot = status_pivot.reindex(index=all_models, columns=SYSTEM_ORDER)
    label_pivot  = label_pivot.reindex(index=all_models, columns=SYSTEM_ORDER)

    # Sort models by median ms/step across ok cells (fastest first)
    median_ms = ms_pivot.median(axis=1)
    order = median_ms.sort_values().index
    return (
        ms_pivot.loc[order],
        status_pivot.loc[order],
        label_pivot.loc[order],
    )


def main():
    df = load_data()
    ms_piv, status_piv, label_piv = build_pivots(df)

    models  = ms_piv.index.tolist()
    systems = ms_piv.columns.tolist()
    n_rows, n_cols = len(models), len(systems)

    data = ms_piv.values.astype(float)

    # Log colour normalisation over ok values only
    ok_vals = data[~np.isnan(data)]
    vmin, vmax = ok_vals.min(), ok_vals.max()
    norm = mcolors.LogNorm(vmin=max(vmin, 0.1), vmax=vmax)
    cmap = mcm.RdYlGn_r   # green = fast, red = slow

    fig_w = 2.2 + n_cols * 1.05
    fig_h = 1.0 + n_rows * 0.52
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Background: grey for all cells, then colour ok cells on top
    grey_bg = np.zeros((n_rows, n_cols, 4))
    grey_bg[:, :, :] = mcolors.to_rgba("#d0d0d0")
    ax.imshow(grey_bg, aspect="auto", interpolation="none")

    # Colour ok cells
    rgba = np.full((n_rows, n_cols, 4), np.nan)
    for r in range(n_rows):
        for c in range(n_cols):
            v = data[r, c]
            if not np.isnan(v):
                rgba[r, c] = cmap(norm(v))

    # Mask nans so imshow skips them
    masked = np.ma.masked_invalid(rgba.mean(axis=2))   # just for extent; we draw rgba directly
    im = ax.imshow(rgba, aspect="auto", interpolation="none")

    # Fake mappable for colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Cell annotations
    for r in range(n_rows):
        for c in range(n_cols):
            v = data[r, c]
            if not np.isnan(v):
                # Determine text colour from luminance
                bg_rgba = cmap(norm(v))
                lum = 0.299 * bg_rgba[0] + 0.587 * bg_rgba[1] + 0.114 * bg_rgba[2]
                tc = "white" if lum < 0.45 else "black"
                if v >= 1000:
                    label = f"{v/1000:.1f}s"
                elif v >= 100:
                    label = f"{v:.0f}ms"
                else:
                    label = f"{v:.1f}ms"
                ax.text(c, r, label, ha="center", va="center", fontsize=7.5, color=tc)
            else:
                fail = label_piv.iloc[r, c]
                fail = fail if isinstance(fail, str) else ""
                ax.text(c, r, fail, ha="center", va="center",
                        fontsize=7, color="#555555", style="italic")

    # Axes
    x_labels = [f"{s}\n({N_ATOMS[s]:,} atoms)" for s in systems]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    y_labels = [PRETTY_MODEL.get(m, m) for m in models]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)

    ax.set_title(
        "MD speed (ms / step) — g7e.4xlarge\n"
        "Models sorted by median speed across systems (fastest → slowest)",
        fontsize=10, fontweight="bold", pad=42,
    )

    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("ms / step  (log scale)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(
        0.5, -0.01,
        "Grey cells: skip = not attempted, OOM = out of memory, err = other error, t/o = timeout, killed = process killed.",
        ha="center", fontsize=7.5, color="gray",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
