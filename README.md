# MLIP Benchmark

Speed and accuracy benchmark of foundational machine learning interatomic potentials (MLIPs) for biomolecular simulation.

**[→ Interactive results](https://frgoe003.github.io/cheminformer/)**

## What's benchmarked

**Accuracy** — mean absolute error on the [SPICE v3](https://zenodo.org/records/19633352) test set (800 structures across small ligands, large ligands, pentapeptides, and dimers).

**Speed** — NVT MD throughput (ms/step, ns/day) on 10 systems ranging from 22 to 100k atoms, measured on three AWS GPU instances: `g7e.4xlarge` (RTX PRO 6000 Blackwell), `g6e.4xlarge` (L40S), and `g5.4xlarge` (A10G).

| Model family | Models |
|---|---|
| MACE | MACE-OFF23(S/L), MACE-OFF24(M), MACE-MH-1, MACE-OMOL-0, Polar-1(S/M/L) |
| AceFF | AceFF-1.1, AceFF-2.0 |
| FeNNix | FeNNix-Bio1(S/M) |
| UMA | UMA-s-1, UMA-m-1 |
| Orb | Orb-v3-omol |
| AIMNet2 | AIMNet2 |
| Egret | Egret-1 |
| MACELES | MACELES-OFF |

## Running the benchmark

```bash
# Set up conda environments (Python 3.11, PyTorch cu128)
bash benchmark/setup.sh

# Run a model (example)
conda activate ase-mace
python benchmark/run_mace.py
```