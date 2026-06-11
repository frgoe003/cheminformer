#!/usr/bin/env bash
# Run all model categories in their dedicated conda environments, then merge results.
# Extra args (e.g. --n-steps 100 --spice-mae) are forwarded to every script.
#
# Usage:
#   ./run_all.sh
#   ./run_all.sh --n-warmup 5 --n-steps 30
#   ./run_all.sh --spice-mae --systems capped_ala chignolin ubiquitin

set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/opt/dlami/nvme"
CONDA_DIR="$PROJECT_ROOT/miniconda3"
source "$CONDA_DIR/etc/profile.d/conda.sh"

cd "$BENCHMARK_DIR"

run_category() {
    local env="$1"
    local script="$2"
    local out="$3"
    shift 3
    if [ -f "$BENCHMARK_DIR/$out" ]; then
        echo "SKIP  $script  (already have $out)"
        return 0
    fi
    echo ""
    echo "════════════════════════════════════════"
    echo "  $env  →  $script"
    echo "════════════════════════════════════════"
    conda run -n "$env" python "$script" "$@"
}

# ── Per-category runs ─────────────────────────────────────────────────────────
run_category ase-mace     run_mace.py     results_mace.csv     "$@"
run_category ase-maceles  run_maceles.py  results_maceles.csv  "$@"
run_category ase-egret    run_egret.py    results_egret.csv    "$@"
run_category ase-aceff    run_aceff.py    results_aceff.csv    "$@"
run_category ase-uma      run_uma.py      results_uma.csv      "$@"
run_category ase-aimnet   run_aimnet.py   results_aimnet.csv   "$@"
run_category ase-fennix   run_fennix.py   results_fennix.csv   "$@"
run_category ase-orb      run_orb.py      results_orb.csv      "$@"
run_category ase-openmm   run_openmm.py      results_openmm.csv      "$@"
run_category openmm-ml    run_openmm_ml.py   results_openmm_ml.csv   "$@"

# ── Merge MD results ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  Merging results"
echo "════════════════════════════════════════"

python3 - "$BENCHMARK_DIR" <<'PYEOF'
import csv, pathlib, sys

bench = pathlib.Path(sys.argv[1])

# MD results
md_files = sorted(bench.glob("results_*.csv"))
md_files = [f for f in md_files if f.name != "results_all.csv"]

if md_files:
    header = None
    rows = []
    for f in md_files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            if header is None:
                header = reader.fieldnames
            rows.extend(reader)
    out = bench / "results_all.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"MD results:  {len(rows)} rows from {len(md_files)} files  →  {out}")
else:
    print("No per-category MD result files found.")

# MAE results
mae_files = sorted(bench.glob("mae_*.csv"))
mae_files = [f for f in mae_files if f.name != "mae_all.csv"]

if mae_files:
    header = None
    rows = []
    for f in mae_files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            if header is None:
                header = reader.fieldnames
            rows.extend(reader)
    out = bench / "mae_all.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"MAE results: {len(rows)} rows from {len(mae_files)} files  →  {out}")
PYEOF

echo ""
echo "All done."
