#!/usr/bin/env python
"""
Multi-model throughput benchmark.

Sweeps system × model × dtype × device and reports steps/second.
Each combination runs in an isolated spawned process so an OOM kill
cannot take down the whole sweep.  Results are saved to benchmark/results.csv.

Usage:
    python run_benchmark.py                          # all combinations
    python run_benchmark.py --devices cpu            # CPU only
    python run_benchmark.py --systems capped_ala chignolin --models 'MACE-OFF23(S)'
    python run_benchmark.py --n-warmup 20 --n-steps 100 --timeout 300
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
_SUPPRESS = (
    "does not match model dtype",
    "Using float32 for MACECalculator",
)
logging.getLogger().addFilter(
    type("_NoiseFilter", (logging.Filter,), {
        "filter": staticmethod(lambda r: not any(s in r.getMessage() for s in _SUPPRESS))
    })()
)

import numpy as np
import torch
from ase import Atoms, constraints, units
from ase.io import read as ase_read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

def _has_element_col(path: Path) -> bool:
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                return len(line.rstrip("\n")) >= 78 and line[76:78].strip() != ""
    return False

def _sym(name: str) -> str:
    s = re.sub(r"^\d+", "", name.strip())
    for prefix in ("CA", "CB", "CH", "C"):
        if s.startswith(prefix):
            return "C"
    if s.startswith("N"): return "N"
    if s.startswith("O"): return "O"
    if s.startswith("H"): return "H"
    return s[0].upper()

def read_pdb_vacuum(path: Path) -> Atoms:
    """Fallback reader for PDBs with non-standard atom names."""
    syms, pos = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                syms.append(_sym(line[12:16]))
                pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return Atoms(symbols=syms, positions=pos, pbc=False)

PROJECT     = Path(__file__).resolve().parent
SYSTEMS_DIR = PROJECT / "systems"
SPICE_HDF5  = PROJECT / "SPICE-test.hdf5"

SYSTEMS = [
    dict(name="capped_ala",   pdb=SYSTEMS_DIR / "capped_ala_22.pdb",    charge=0, spin=1),
    dict(name="chignolin",    pdb=SYSTEMS_DIR / "chignolin_138.pdb",    charge=0, spin=1),
    dict(name="ubiquitin",    pdb=SYSTEMS_DIR / "ubiquitin_602.pdb",    charge=0, spin=1),
    dict(name="2LZM",         pdb=SYSTEMS_DIR / "2LZM.pdb",             charge=0, spin=1),
    dict(name="1ZG4",         pdb=SYSTEMS_DIR / "1ZG4_2p2k.pdb",        charge=0, spin=1),
    dict(name="3N5G",         pdb=SYSTEMS_DIR / "3N5G.pdb",             charge=0, spin=1),
    dict(name="5G1P",         pdb=SYSTEMS_DIR / "5G1P_13k.pdb",         charge=0, spin=1),
    dict(name="1B3B",         pdb=SYSTEMS_DIR / "1B3B_19k.pdb",         charge=0, spin=1),
    dict(name="9VM6",         pdb=SYSTEMS_DIR / "9VM6.pdb",             charge=0, spin=1),
]

# ── Model groups ──────────────────────────────────────────────────────────────
POLAR_MODELS     = ["polar-1-s", "polar-1-m", "polar-1-l"]
ACEFF_MODELS     = ["AceFF-1.1", "AceFF-2.0"]
AIMNET_MODELS    = ["AIMNet2"]
EGRET_MODELS     = ["Egret-1"]
FENNIX_MODELS    = ["FeNNix-Bio1(S)", "FeNNix-Bio1(M)"]
MACE_MH_MODELS   = ["MACE-MH-1"]
MACE_OFF_MODELS  = ["MACE-OFF23(S)", "MACE-OFF23(L)", "MACE-OFF24(M)", "MACELES-OFF"]
MACE_OMOL_MODELS = ["MACE-OMOL-0"]
ORB_MODELS       = ["Orb-v3-omol"]
UMA_MODELS       = ["UMA-s-1", "UMA-m-1"]

ALL_MODELS = (
    POLAR_MODELS + ACEFF_MODELS + AIMNET_MODELS + EGRET_MODELS +
    FENNIX_MODELS + MACE_MH_MODELS + MACE_OFF_MODELS +
    MACE_OMOL_MODELS + ORB_MODELS + UMA_MODELS
)
ALL_DTYPES = ["float32"]

RESULT_FIELDS = [
    "system", "n_atoms", "model", "dtype", "device",
    "n_warmup", "n_steps", "elapsed_s", "steps_per_s", "ms_per_step",
    "status", "error",
]

MAE_RESULT_FIELDS = [
    "model", "dtype", "device", "n_molecules", "mae_ev", "mae_kcal_mol",
    "status", "error",
]

_HARTREE_TO_EV = 27.211386245988

# ── Element support ───────────────────────────────────────────────────────────
_ORGANIC_Z = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})        # H C N O F P S Cl Br I
_AIMNET2_Z = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53})  # + B Si As Se
_FENNIX_Z  = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53})          # + B Si
_ALL_Z     = frozenset(range(1, 119))

def supported_elements(model_name: str) -> frozenset:
    if model_name in MACE_OFF_MODELS or model_name in EGRET_MODELS or model_name in POLAR_MODELS:
        return _ORGANIC_Z
    if model_name in ACEFF_MODELS:
        return _ORGANIC_Z
    if model_name in AIMNET_MODELS:
        return _AIMNET2_Z
    if model_name in FENNIX_MODELS:
        return _FENNIX_Z
    # MACE-MH, MACE-OMOL, Orb, UMA: broad coverage
    return _ALL_Z

# ── Per-model lookup tables ───────────────────────────────────────────────────
_MACE_OFF_IDS = {
    "MACE-OFF23(S)": "small",
    "MACE-OFF23(L)": "large",
    "MACE-OFF24(M)": "https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true",
    "MACELES-OFF":   "https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/MACELES-OFF_small_converted.model?raw=true",
    "Egret-1":       "https://github.com/rowansci/egret-public/blob/master/compiled_models/EGRET_1.model?raw=true",
}

_MACE_MH_URLS = {
    "MACE-MH-1": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
}

_ACEFF_CONFIGS = {
    # (hf_repo_id, filename, coulomb_cutoff or None)
    "AceFF-1.1": ("Acellera/AceFF-1.1", "aceff_v1.1.ckpt", None),
    "AceFF-2.0": ("Acellera/AceFF-2.0", "aceff_v2.0.ckpt", 10.0),
}

_UMA_FILES = {
    "UMA-s-1": PROJECT / "uma-s-1p2.pt",
    "UMA-m-1": PROJECT / "uma-m-1p1.pt",
}

_FENNIX_FILES = {
    "FeNNix-Bio1(S)": "fennix-bio1S.fnx",
    "FeNNix-Bio1(M)": "fennix-bio1M.fnx",
}


# ── Calculator factory ────────────────────────────────────────────────────────
def create_calculator(model_name: str, device: str, dtype: str):
    if model_name in POLAR_MODELS:
        from mace.calculators import mace_polar
        return mace_polar(model=model_name, device=device,
                          default_dtype=dtype, enable_cueq=False)

    if model_name in MACE_OFF_MODELS or model_name in EGRET_MODELS:
        from mace.calculators.foundations_models import mace_off
        return mace_off(_MACE_OFF_IDS[model_name], default_dtype=dtype, device=device)

    if model_name in MACE_MH_MODELS:
        from mace.calculators.foundations_models import mace_mp
        return mace_mp(_MACE_MH_URLS[model_name], default_dtype=dtype,
                       device=device, head="spice_wB97M")

    if model_name in MACE_OMOL_MODELS:
        from mace.calculators.foundations_models import mace_omol
        return mace_omol("extra_large", default_dtype=dtype, device=device)

    if model_name in ACEFF_MODELS:
        from huggingface_hub import hf_hub_download
        from torchmdnet.calculators import TMDNETCalculator
        repo_id, filename, coulomb_cutoff = _ACEFF_CONFIGS[model_name]
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        kwargs = {"device": device}
        if coulomb_cutoff is not None:
            kwargs["coulomb_cutoff"] = coulomb_cutoff
        return TMDNETCalculator(path, **kwargs)

    if model_name in AIMNET_MODELS:
        from aimnet.calculators import AIMNet2Calculator
        return AIMNet2Calculator("aimnet2")

    if model_name in ORB_MODELS:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        orbff, atoms_adapter = pretrained.orb_v3_conservative_omol(
            device=device, precision=f"{dtype}-high"
        )
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)

    if model_name in UMA_MODELS:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        predictor = load_predict_unit(
            path=_UMA_FILES[model_name], device=device, inference_settings="default"
        )
        return FAIRChemCalculator(predictor, task_name="omol")

    if model_name in FENNIX_MODELS:
        from fennol.ase import FENNIXCalculator
        return FENNIXCalculator(model=_FENNIX_FILES[model_name],
                                matmul_prec="highest", gpu_preprocessing=True)

    raise ValueError(f"Unknown model: {model_name!r}")


def load_atoms(sys_def: dict):
    pdb = Path(sys_def["pdb"])
    if _has_element_col(pdb):
        atoms = ase_read(str(pdb))
        atoms.pbc = False
    else:
        atoms = read_pdb_vacuum(pdb)
    atoms.info["charge"] = sys_def["charge"]
    atoms.info["spin"]   = sys_def["spin"]
    return atoms


def _worker(queue, system_name, model_name, dtype, device, n_warmup, n_steps):
    """Runs in a spawned child process; puts a result dict into queue."""
    try:
        sys_def = next(s for s in SYSTEMS if s["name"] == system_name)

        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        atoms = load_atoms(sys_def)

        if model_name in POLAR_MODELS:
            atoms.info["external_field"] = [0.0, 0.0, 0.0]

        calc  = create_calculator(model_name, device, dtype)
        atoms.calc = calc

        if model_name in AIMNET_MODELS:
            calc.set_charge(sys_def["charge"])
            calc.set_mult(sys_def["spin"])
        elif model_name in FENNIX_MODELS:
            charges = [0] * len(atoms)
            charges[0] = sys_def["charge"]
            atoms.set_initial_charges(charges)

        MaxwellBoltzmannDistribution(atoms, temperature_K=300.0,
                                     rng=np.random.default_rng(42))
        atoms.set_constraint(constraints.FixCom())

        dyn = Langevin(atoms, timestep=units.fs, temperature_K=300.0,
                       friction=0.01 / units.fs, fixcm=False)

        # ── Warmup (not timed, trajectory written here for validation) ────────
        device_str = device.replace(":", "")
        safe_name  = re.sub(r"[^\w.-]", "_", model_name)
        traj_path  = (PROJECT / "benchmark" / "trajectories" /
                      f"{system_name}_{safe_name}_{dtype}_{device_str}.xyz")
        traj_path.parent.mkdir(parents=True, exist_ok=True)

        dyn.attach(lambda: write(str(traj_path), atoms, append=True), interval=10)
        dyn.run(n_warmup)

        # ── Timed run (pure compute, no extra I/O) ────────────────────────────
        dyn.observers.clear()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dyn.run(n_steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        queue.put({
            "system":      system_name,
            "n_atoms":     len(atoms),
            "elapsed_s":   round(elapsed, 4),
            "steps_per_s": round(n_steps / elapsed, 2),
            "ms_per_step": round(1000 * elapsed / n_steps, 2),
            "status":      "ok",
            "error":       "",
        })
    except Exception:
        queue.put({
            "status": "error",
            "error":  traceback.format_exc().splitlines()[-1],
        })


def run_isolated(system_name, model_name, dtype, device,
                 n_warmup, n_steps, timeout) -> dict:
    """Spawns a fresh process per run; survives OOM kills and timeouts."""
    ctx   = mp.get_context("spawn")
    queue = ctx.Queue()
    proc  = ctx.Process(
        target=_worker,
        args=(queue, system_name, model_name, dtype, device, n_warmup, n_steps),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"status": "timeout", "error": f">{timeout}s"}

    if proc.exitcode == -9:
        return {"status": "killed", "error": "OOM/SIGKILL"}

    if not queue.empty():
        return queue.get_nowait()

    return {"status": "error", "error": f"exit code {proc.exitcode}"}


def _mae_worker(queue, model_name, device, dtype, hdf5_path):
    """Runs in a spawned child process; evaluates single-point energies on all SPICE molecules."""
    try:
        import h5py

        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        calc = create_calculator(model_name, device, dtype)

        if model_name in AIMNET_MODELS:
            calc.set_charge(0)
            calc.set_mult(1)

        mol_maes = []

        with h5py.File(hdf5_path, "r") as f:
            for mol_data in f.values():
                atomic_numbers = mol_data["atomic_numbers"][:]
                conformations  = mol_data["conformations"][:]
                dft_energies   = mol_data["dft_total_energy"][:] * _HARTREE_TO_EV

                n_conf = len(conformations)
                model_energies = np.empty(n_conf)

                for k, conf in enumerate(conformations):
                    atoms = Atoms(numbers=atomic_numbers, positions=conf, pbc=False)
                    atoms.info["charge"] = 0
                    atoms.info["spin"]   = 1
                    if model_name in POLAR_MODELS:
                        atoms.info["external_field"] = [0.0, 0.0, 0.0]
                    if model_name in FENNIX_MODELS:
                        atoms.set_initial_charges([0] * len(atoms))
                    atoms.calc = calc
                    model_energies[k] = atoms.get_potential_energy()

                i_idx, j_idx = np.triu_indices(n_conf, k=1)
                mol_mae = np.mean(np.abs(
                    (model_energies[i_idx] - model_energies[j_idx]) -
                    (dft_energies[i_idx]   - dft_energies[j_idx])
                ))
                mol_maes.append(float(mol_mae))

        mean_mae_ev = float(np.mean(mol_maes))
        queue.put({
            "status":        "ok",
            "n_molecules":   len(mol_maes),
            "mae_ev":        round(mean_mae_ev, 6),
            "mae_kcal_mol":  round(mean_mae_ev * 23.0605, 4),
        })
    except Exception:
        queue.put({
            "status": "error",
            "error":  traceback.format_exc().splitlines()[-1],
        })


def run_mae_isolated(model_name, device, dtype, timeout) -> dict:
    """Spawns a fresh process for the full SPICE MAE evaluation of one model."""
    ctx   = mp.get_context("spawn")
    queue = ctx.Queue()
    proc  = ctx.Process(
        target=_mae_worker,
        args=(queue, model_name, device, dtype, str(SPICE_HDF5)),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"status": "timeout", "error": f">{timeout}s"}

    if proc.exitcode == -9:
        return {"status": "killed", "error": "OOM/SIGKILL"}

    if not queue.empty():
        return queue.get_nowait()

    return {"status": "error", "error": f"exit code {proc.exitcode}"}


def collect_system_info() -> dict:
    info: dict = {}

    # ── Software ──────────────────────────────────────────────────────────────
    info["python"]     = sys.version.split()[0]
    info["torch"]      = torch.__version__
    info["torch_cuda"] = torch.version.cuda or "N/A"

    # ── GPU (torch + nvidia-smi) ──────────────────────────────────────────────
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"]               = props.name
        info["gpu_vram_gib"]           = round(props.total_memory / 1024**3, 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_sm_count"]           = props.multi_processor_count
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=driver_version,clocks.max.sm,clocks.max.memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                if len(parts) >= 3:
                    info["gpu_driver"]            = parts[0]
                    info["gpu_max_sm_clock_mhz"]  = parts[1]
                    info["gpu_max_mem_clock_mhz"] = parts[2]
        except Exception:
            pass
    else:
        info["gpu_name"] = "none"

    # ── CPU ───────────────────────────────────────────────────────────────────
    info["cpu_logical_cores"] = os.cpu_count()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if "model name" in line:
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        pairs = set(zip(
            re.findall(r"physical id\s*:\s*(\d+)", cpuinfo),
            re.findall(r"core id\s*:\s*(\d+)", cpuinfo),
        ))
        if pairs:
            info["cpu_physical_cores"] = len(pairs)
    except Exception:
        pass
    try:
        freq = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read_text()
        info["cpu_max_freq_mhz"] = round(int(freq.strip()) / 1000, 1)
    except Exception:
        pass

    # ── RAM ───────────────────────────────────────────────────────────────────
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["ram_total_gib"] = round(int(line.split()[1]) / 1024**2, 2)
            elif line.startswith("MemAvailable:"):
                info["ram_available_gib"] = round(int(line.split()[1]) / 1024**2, 2)
    except Exception:
        pass

    return info


def main():
    p = argparse.ArgumentParser(description="Multi-model throughput benchmark")
    p.add_argument("--devices",  nargs="+", default=None,
                   help="Devices to test (default: cuda:0 if available + cpu)")
    p.add_argument("--models",   nargs="+", default=ALL_MODELS,
                   help=f"Models to benchmark (default: all). Available: {ALL_MODELS}")
    p.add_argument("--dtypes",   nargs="+", default=ALL_DTYPES, choices=ALL_DTYPES)
    p.add_argument("--systems",  nargs="+", default=None,
                   help="System names: capped_ala chignolin ubiquitin ...")
    p.add_argument("--n-warmup", type=int, default=10,
                   help="Warmup steps discarded before timing (default: 10)")
    p.add_argument("--n-steps",  type=int, default=50,
                   help="Steps timed per run (default: 50)")
    p.add_argument("--max-step-time", type=float, default=10.0,
                   help="Max wall-clock seconds allowed per step; "
                        "total timeout = (n-warmup + n-steps) × this (default: 10)")
    p.add_argument("--out",          default="benchmark/results.csv")
    p.add_argument("--spice-mae",    action="store_true",
                   help="Run SPICE test-set energy MAE benchmark after the MD sweep")
    p.add_argument("--mae-timeout",  type=int, default=7200,
                   help="Timeout in seconds for the MAE run per model (default: 7200)")
    args = p.parse_args()

    unknown = [m for m in args.models if m not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Available: {ALL_MODELS}")

    if args.devices:
        devices = args.devices
    else:
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA device found. Pass --devices cpu to run on CPU.")
        devices = ["cuda:0"]

    systems = SYSTEMS
    if args.systems:
        names = set(args.systems)
        systems = [s for s in SYSTEMS if s["name"] in names]
        if not systems:
            raise SystemExit(f"No matching systems. Available: {[s['name'] for s in SYSTEMS]}")

    for s in systems:
        if not Path(s["pdb"]).exists():
            raise SystemExit(f"Missing: {s['pdb']}")

    # Precompute element sets so we can skip incompatible model/system pairs
    system_elements = {s["name"]: frozenset(load_atoms(s).get_atomic_numbers()) for s in systems}

    timeout = int((args.n_warmup + args.n_steps) * args.max_step_time)

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total   = len(devices) * len(args.models) * len(args.dtypes) * len(systems)
    done    = 0
    results: list[dict] = []

    for device in devices:
        for model_name in args.models:
            for dtype in args.dtypes:
                for sys_def in systems:
                    done += 1
                    tag = (f"[{done}/{total}]  {sys_def['name']:<12}"
                           f"  {model_name:<18}  {dtype:<8}  {device}")
                    print(tag, end="  ", flush=True)

                    unsupported = system_elements[sys_def["name"]] - supported_elements(model_name)
                    if unsupported:
                        from ase.data import chemical_symbols
                        syms = ", ".join(chemical_symbols[z] for z in sorted(unsupported))
                        print(f"SKIP (unsupported elements: {syms})")
                        row = {"status": "skipped", "error": f"unsupported elements: {syms}"}
                    else:
                        row = run_isolated(
                            sys_def["name"], model_name, dtype, device,
                            args.n_warmup, args.n_steps, timeout,
                        )

                        if row.get("status") == "ok":
                            print(f"{row['steps_per_s']:>8.2f} steps/s"
                                  f"  ({row['ms_per_step']:.2f} ms/step)")
                        else:
                            print(f"{row['status'].upper()}: {row['error']}")

                    row.setdefault("system",      sys_def["name"])
                    row.setdefault("n_atoms",     "?")
                    row.setdefault("model",       model_name)
                    row.setdefault("dtype",       dtype)
                    row.setdefault("device",      device)
                    row.setdefault("n_warmup",    args.n_warmup)
                    row.setdefault("n_steps",     args.n_steps)
                    row.setdefault("elapsed_s",   "")
                    row.setdefault("steps_per_s", "")
                    row.setdefault("ms_per_step", "")
                    results.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    n_ok  = sum(1 for r in results if r.get("status") == "ok")
    n_err = len(results) - n_ok
    print(f"\nDone. {n_ok} ok, {n_err} failed.")
    print(f"Results saved → {out_path}")

    sys_info = collect_system_info()
    sys_info_path = out_path.with_name("system_info.json")
    with open(sys_info_path, "w") as f:
        json.dump(sys_info, f, indent=2)
    print(f"System info  → {sys_info_path}")

    # ── SPICE energy MAE benchmark ────────────────────────────────────────────
    if args.spice_mae:
        if not SPICE_HDF5.exists():
            print(f"\nSPICE HDF5 not found at {SPICE_HDF5}, skipping MAE benchmark.")
        else:
            total_mae = len(devices) * len(args.models) * len(args.dtypes)
            done_mae  = 0
            mae_results: list[dict] = []

            print(f"\n── SPICE energy MAE  ({total_mae} runs) ──")
            for device in devices:
                for model_name in args.models:
                    for dtype in args.dtypes:
                        done_mae += 1
                        print(f"[{done_mae}/{total_mae}]  {model_name:<18}  {dtype}  {device}  ",
                              end="", flush=True)

                        row = run_mae_isolated(model_name, device, dtype, args.mae_timeout)

                        if row.get("status") == "ok":
                            print(f"MAE = {row['mae_kcal_mol']:.4f} kcal/mol"
                                  f"  ({row['mae_ev']:.6f} eV)"
                                  f"  n={row['n_molecules']}")
                        else:
                            print(f"{row['status'].upper()}: {row.get('error', '')}")

                        row.setdefault("model",        model_name)
                        row.setdefault("dtype",        dtype)
                        row.setdefault("device",       device)
                        row.setdefault("n_molecules",  "")
                        row.setdefault("mae_ev",       "")
                        row.setdefault("mae_kcal_mol", "")
                        row.setdefault("error",        "")
                        mae_results.append(row)

            mae_out = out_path.with_name("mae_results.csv")
            with open(mae_out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MAE_RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(mae_results)

            n_ok  = sum(1 for r in mae_results if r.get("status") == "ok")
            n_err = len(mae_results) - n_ok
            print(f"\nMAE done. {n_ok} ok, {n_err} failed.")
            print(f"MAE results  → {mae_out}")


if __name__ == "__main__":
    main()
