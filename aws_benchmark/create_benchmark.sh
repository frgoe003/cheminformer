#!/usr/bin/env bash
set -euo pipefail

# Ensure conda is available in this shell (needed after a fresh login post-setup.sh)
if ! command -v conda &> /dev/null; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLS_DIR="$SCRIPT_DIR/mols"
SYSTEMS_DIR="$SCRIPT_DIR/benchmark/systems"
ENV_NAME="ase"

echo "[INFO] Preparing benchmark systems in $SYSTEMS_DIR"
mkdir -p "$SYSTEMS_DIR"

# ── capped alanine (22 atoms, custom atom names) ──────────────────────────────
cp "$MOLS_DIR/capped_ala.pdb" "$SYSTEMS_DIR/capped_ala.pdb"
N=$(grep -c '^ATOM\|^HETATM' "$SYSTEMS_DIR/capped_ala.pdb")
echo "[OK]  capped_ala.pdb   ($N atoms)"

# ── chignolin (138 atoms, already protonated) ─────────────────────────────────
cp "$MOLS_DIR/chignolin.pdb" "$SYSTEMS_DIR/chignolin.pdb"
N=$(grep -c '^ATOM\|^HETATM' "$SYSTEMS_DIR/chignolin.pdb")
echo "[OK]  chignolin.pdb    ($N atoms)"

# ── ubiquitin: add H atoms via pdbfixer ───────────────────────────────────────
echo "[INFO] Installing pdbfixer + openmm …"
conda run -n "$ENV_NAME" pip install pdbfixer openmm -q

echo "[INFO] Adding H atoms to ubiquitin (pH 7.0) …"
conda run -n "$ENV_NAME" python - \
    "$MOLS_DIR/ubiquitin.pdb" \
    "$SYSTEMS_DIR/ubiquitin_H.pdb" \
<< 'PYEOF'
import sys
from pdbfixer import PDBFixer
from openmm.app import PDBFile

src, dst = sys.argv[1], sys.argv[2]
fixer = PDBFixer(filename=src)
fixer.removeHeterogens(keepWater=False)
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)
with open(dst, "w") as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)
n = sum(1 for line in open(dst) if line.startswith(('ATOM', 'HETATM')))
print(f"[OK]  ubiquitin_H.pdb  ({n} atoms)")
PYEOF

echo ""
echo "[INFO] Benchmark systems ready:"
for f in "$SYSTEMS_DIR"/*.pdb; do
    n=$(grep -c '^ATOM\|^HETATM' "$f")
    printf "  %-30s %d atoms\n" "$(basename "$f")" "$n"
done
