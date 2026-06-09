#!/usr/bin/env bash
# setup_new.sh — reproducible environment setup for all NNP benchmark envs.
#
# Tested on: NVIDIA RTX PRO 6000 Blackwell (sm_120), CUDA 12.9, Driver 13.2
# All envs use Python 3.11 and CUDA 12.8 (cu128) PyTorch wheels so that the
# Blackwell GPU is fully supported.
#
# Usage:  bash setup_new.sh
#   Re-running is safe: existing environments are removed and rebuilt cleanly.

set -euo pipefail

###############################################################################
# Config
###############################################################################

PROJECT_ROOT="/opt/dlami/nvme"
BENCHMARK_DIR="$PROJECT_ROOT/benchmark"
SRC_DIR="$PROJECT_ROOT/src"
CONDA_DIR="$PROJECT_ROOT/miniconda3"
PYTHON_VERSION="3.11"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

###############################################################################
# 1. Install / locate Miniconda
###############################################################################

if [ ! -d "$CONDA_DIR" ]; then
    echo "[INFO] Installing Miniconda → $CONDA_DIR"
    INSTALLER="/tmp/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$CONDA_DIR"
    rm -f "$INSTALLER"
else
    echo "[INFO] Miniconda already at $CONDA_DIR"
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

if ! grep -qF "$CONDA_DIR/etc/profile.d/conda.sh" "$HOME/.bashrc" 2>/dev/null; then
    printf '\n. "%s/etc/profile.d/conda.sh"\n' "$CONDA_DIR" >> "$HOME/.bashrc"
fi

###############################################################################
# 2. Clone shared source repositories
###############################################################################

mkdir -p "$SRC_DIR"

clone_or_update() {
    local url="$1" dest="$2"
    if [ ! -d "$dest" ]; then
        echo "[INFO] Cloning $url → $dest"
        git clone "$url" "$dest"
    else
        echo "[INFO] Repo already present: $dest"
    fi
}

clone_or_update "https://github.com/ACEsuit/mace.git"                       "$SRC_DIR/mace"
clone_or_update "https://github.com/WillBaldwin0/graph_electrostatics.git"  "$SRC_DIR/graph_electrostatics"

###############################################################################
# Helper: (re)create a clean conda env with pip upgraded
###############################################################################

make_env() {
    local name="$1"
    echo ""
    echo "====== $name ======"
    if conda env list | grep -qE "^$name\s"; then
        echo "[INFO] Removing existing env $name"
        conda remove -n "$name" --all -y -q
    fi
    conda create -n "$name" python="$PYTHON_VERSION" -y -q
    conda run -n "$name" pip install --upgrade pip -q
}

###############################################################################
# Helper: install torch + torchvision + torchaudio from the cu128 wheel index
###############################################################################

install_torch() {
    local name="$1"
    echo "[INFO] $name: installing PyTorch (cu128)"
    conda run -n "$name" pip install -q \
        torch torchvision torchaudio \
        --index-url "$TORCH_INDEX"
}

###############################################################################
# Patch helper — inline Python applied after torch is installed.
#
# patch_torch ENV
#   Applies two defensive fixes that are needed when the env has both
#   pytorch_lightning (→ torchmetrics → torchvision) and the Blackwell GPU
#   prevents loading the torchvision C extension:
#
#   (a) torchvision/_meta_registrations.py – guards the register_meta decorator
#       against the circular-import AttributeError that fires when
#       torchvision.extension is not yet set on the partially-initialised
#       torchvision module.
#
#   (b) torch/_library/fake_impl.py – makes register_fake silently skip
#       registration for operators that don't exist (e.g. torchvision::nms
#       when the CUDA extension failed to load), instead of raising RuntimeError.
###############################################################################

patch_torch() {
    local name="$1"
    echo "[INFO] $name: applying torch/torchvision patches"
    conda run -n "$name" python - << 'PYEOF'
import sys
from pathlib import Path

sp = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

# ── (a) torchvision/_meta_registrations.py ─────────────────────────────────
tv_meta = sp / "torchvision" / "_meta_registrations.py"
if tv_meta.exists():
    src = tv_meta.read_text()
    OLD = "        if torchvision.extension._has_ops():"
    NEW = "        _ext = getattr(torchvision, \"extension\", None)\n        if _ext is not None and _ext._has_ops():"
    if OLD in src:
        tv_meta.write_text(src.replace(OLD, NEW))
        print(f"  patched {tv_meta.relative_to(sp)}")
    else:
        print(f"  already patched: {tv_meta.relative_to(sp)}")
else:
    print("  torchvision not installed — skipping _meta_registrations patch")

# ── (b) torch/_library/fake_impl.py ────────────────────────────────────────
fake = sp / "torch" / "_library" / "fake_impl.py"
if fake.exists():
    src = fake.read_text()
    OLD = (
        "        if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, \"Meta\"):\n"
        "            raise RuntimeError(\n"
        "                f\"register_fake(...): the operator {self.qualname} \"\n"
        "                f\"already has an DispatchKey::Meta implementation via a \"\n"
        "                f\"pre-existing torch.library or TORCH_LIBRARY registration. \"\n"
        "                f\"Please either remove that registration or don't call \"\n"
        "                f\"register_fake.\"\n"
        "            )"
    )
    NEW = (
        "        try:\n"
        "            _has_meta = torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, \"Meta\")\n"
        "        except RuntimeError as _e:\n"
        "            if \"does not exist\" in str(_e):\n"
        "                return RegistrationHandle(lambda: None)\n"
        "            raise\n"
        "        if _has_meta:\n"
        "            raise RuntimeError(\n"
        "                f\"register_fake(...): the operator {self.qualname} \"\n"
        "                f\"already has an DispatchKey::Meta implementation via a \"\n"
        "                f\"pre-existing torch.library or TORCH_LIBRARY registration. \"\n"
        "                f\"Please either remove that registration or don't call \"\n"
        "                f\"register_fake.\"\n"
        "            )"
    )
    if OLD in src:
        fake.write_text(src.replace(OLD, NEW))
        print(f"  patched {fake.relative_to(sp)}")
    elif "does not exist" in src:
        print(f"  already patched: {fake.relative_to(sp)}")
    else:
        print(f"  WARNING: unexpected fake_impl.py content — patch not applied")
PYEOF
}

###############################################################################
# 3. ase-mace  (MACE-POLAR, MACE-OFF23/24, MACE-MH-1, MACE-OMOL-0)
###############################################################################

make_env "ase-mace"
install_torch "ase-mace"
conda install -n "ase-mace" -c conda-forge openff-toolkit -y -q
conda run -n "ase-mace" pip install -q \
    mace-torch \
    cuequivariance_torch \
    -e "$SRC_DIR/graph_electrostatics"
echo "[INFO] ase-mace: done"

###############################################################################
# 4. ase-maceles  (MACELES-OFF)
###############################################################################

make_env "ase-maceles"
install_torch "ase-maceles"
conda run -n "ase-maceles" pip install -q \
    mace-torch \
    cuequivariance_torch \
    "git+https://github.com/ChengUCB/les.git"
echo "[INFO] ase-maceles: done"

###############################################################################
# 5. ase-egret  (Egret-1)
###############################################################################

make_env "ase-egret"
install_torch "ase-egret"
conda run -n "ase-egret" pip install -q \
    mace-torch \
    cuequivariance_torch
echo "[INFO] ase-egret: done"

###############################################################################
# 6. ase-aceff  (AceFF-1.1, AceFF-2.0)
###############################################################################

make_env "ase-aceff"
install_torch "ase-aceff"
conda run -n "ase-aceff" pip install -q \
    huggingface_hub torchmd-net rdkit ase
# openff-toolkit is not on PyPI; install via conda-forge then its pure-Python deps.
conda install -n "ase-aceff" -c conda-forge openff-toolkit -y -q
conda run -n "ase-aceff" pip install -q pint cachetools

# aceff_calculator stub — provides ACEFF_ATOMIC_NUMBERS (H B C N O F Si P S Cl Br I).
# The Acellera aceff_calculator package is not publicly distributed on PyPI/conda;
# element coverage taken from the AceFF-1.1 and AceFF-2.0 HuggingFace READMEs.
cat > "$BENCHMARK_DIR/aceff_calculator.py" << 'PYEOF'
# Stub for the private aceff_calculator package.
# Elements from AceFF-1.1/2.0 READMEs: H B C N O F Si P S Cl Br I
ACEFF_ATOMIC_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
PYEOF
echo "[INFO] ase-aceff: done"

###############################################################################
# 7. ase-uma  (UMA-s-1, UMA-m-1)
###############################################################################

make_env "ase-uma"
install_torch "ase-uma"
conda run -n "ase-uma" pip install -q fairchem-core
echo "[INFO] ase-uma: done"

###############################################################################
# 8. ase-aimnet  (AIMNet2)
###############################################################################

make_env "ase-aimnet"
install_torch "ase-aimnet"
conda run -n "ase-aimnet" pip install -q "aimnet[ase]"
# aimnet pulls in pytorch_lightning → torchmetrics → torchvision.
# Apply patches so the Blackwell GPU (sm_120, C extension may not load) doesn't
# cause a circular-import crash when openff.toolkit is later imported.
patch_torch "ase-aimnet"
echo "[INFO] ase-aimnet: done"

###############################################################################
# 9. ase-fennix  (FeNNix-Bio1 S/M)
###############################################################################

make_env "ase-fennix"
install_torch "ase-fennix"
conda run -n "ase-fennix" pip install -q "fennol[cuda]"
echo "[INFO] ase-fennix: done"

###############################################################################
# 10. ase-orb  (Orb-v3-omol)
###############################################################################

make_env "ase-orb"
install_torch "ase-orb"
conda run -n "ase-orb" pip install -q "orb-models" pynanoflann
# orb-models also brings in pytorch_lightning.
patch_torch "ase-orb"
echo "[INFO] ase-orb: done"

###############################################################################
# 11. ase-openmm  (OpenMM-GAFF2, OpenMM-Amber14)
###############################################################################

make_env "ase-openmm"
# OpenMM and openff-toolkit must come from conda-forge.
conda install -n "ase-openmm" -c conda-forge openmm openff-toolkit -y -q
# Force-install cu128 torch after conda packages so the CUDA version is correct.
conda run -n "ase-openmm" pip install -q \
    torch torchvision torchaudio \
    --index-url "$TORCH_INDEX"
conda run -n "ase-openmm" pip install -q openmmforcefields rdkit ase
echo "[INFO] ase-openmm: done"

###############################################################################
# 12. Smoke tests
###############################################################################

echo ""
echo "===== Smoke tests ====="

run_test() {
    local env="$1" code="$2"
    local result
    result=$(conda run -n "$env" python -c "$code" 2>&1 | tail -1)
    echo "  $env: $result"
}

run_test "ase-mace"    "import torch, ase, mace; print('ok  torch=' + torch.__version__)"
run_test "ase-maceles" "import torch, ase, mace; print('ok  torch=' + torch.__version__)"
run_test "ase-egret"   "import torch, ase, mace; print('ok  torch=' + torch.__version__)"
run_test "ase-aceff"   "import torch, ase, torchmdnet; print('ok  torch=' + torch.__version__)"
run_test "ase-uma"     "import torch, ase, fairchem; print('ok  torch=' + torch.__version__)"
run_test "ase-aimnet"  "import torch, ase, aimnet; print('ok  torch=' + torch.__version__)"
run_test "ase-fennix"  "import torch, ase, fennol; print('ok  torch=' + torch.__version__)"
run_test "ase-orb"     "import torch, ase, orb_models; print('ok  torch=' + torch.__version__)"
run_test "ase-openmm"  "import torch, ase, openmm; print('ok  torch=' + torch.__version__)"

echo ""
echo "===== CUDA / GPU check ====="
conda run -n "ase-mace" python -c "
import torch
print(f'  CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU            : {p.name}')
    print(f'  VRAM           : {p.total_memory/1024**3:.1f} GiB')
    print(f'  Compute cap.   : sm_{p.major}{p.minor}')
    print(f'  PyTorch CUDA   : {torch.version.cuda}')
"

echo ""
echo "[INFO] Setup complete."
echo ""
echo "Quick-start examples:"
echo "  conda activate ase-aimnet  && python run_aimnet.py  --spice-mae-only"
echo "  conda activate ase-orb     && python run_orb.py     --spice-mae-only"
echo "  conda activate ase-aceff   && python run_aceff.py   --spice-mae-only"
echo "  conda activate ase-mace    && python run_mace.py    --spice-mae-only"
echo "  conda activate ase-egret   && python run_egret.py   --spice-mae-only"
echo "  conda activate ase-uma     && python run_uma.py     --spice-mae-only"
echo "  conda activate ase-fennix  && python run_fennix.py  --spice-mae-only"
echo "  conda activate ase-maceles && python run_maceles.py --spice-mae-only"
