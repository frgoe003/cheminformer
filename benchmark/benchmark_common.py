"""
Shared infrastructure for per-category benchmark scripts.

Each category script (run_mace.py, run_aceff.py, …) must define at module level:
  - ALL_MODELS: list[str]
  - create_calculator(model_name, device, dtype) -> ASE calculator
  - supported_elements(model_name) -> frozenset[int]
  - configure_atoms(atoms, calc, model_name, charge, spin) [optional]
    Called after atoms.calc is set, for per-model charge/field setup.
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*grad attribute.*non-leaf Tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*global torch default dtype.*")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from ase.optimize import LBFGS

# ── Paths & systems ───────────────────────────────────────────────────────────
PROJECT     = Path(__file__).resolve().parent
SYSTEMS_DIR = PROJECT / "systems"
SPICE_HDF5  = PROJECT / "SPICE-test.hdf5"

SYSTEMS = [
    dict(name="capped_ala", pdb=SYSTEMS_DIR / "capped_ala_22.pdb",  charge=0, spin=1),
    dict(name="chignolin",  pdb=SYSTEMS_DIR / "chignolin_138.pdb",  charge=0, spin=1),
    dict(name="ubiquitin",  pdb=SYSTEMS_DIR / "ubiquitin_602.pdb",  charge=0, spin=1),
    dict(name="2LZM",       pdb=SYSTEMS_DIR / "2LZM.pdb",           charge=0, spin=1),
    dict(name="1ZG4",       pdb=SYSTEMS_DIR / "1ZG4_2p2k.pdb",      charge=0, spin=1),
    dict(name="3N5G",       pdb=SYSTEMS_DIR / "3N5G.pdb",           charge=0, spin=1),
    dict(name="5G1P",       pdb=SYSTEMS_DIR / "5G1P_13k.pdb",       charge=0, spin=1),
    dict(name="1B3B",       pdb=SYSTEMS_DIR / "1B3B_19k.pdb",       charge=0, spin=1),
    dict(name="9VM6",         pdb=SYSTEMS_DIR / "9VM6.pdb",             charge=0, spin=1),
    dict(name="water_99k",    pdb=SYSTEMS_DIR / "water_99k.pdb",        charge=0, spin=1),
    dict(name="water_199k",   pdb=SYSTEMS_DIR / "water_199k.pdb",       charge=0, spin=1),
    dict(name="water_499k",   pdb=SYSTEMS_DIR / "water_499k.pdb",       charge=0, spin=1),
]

# ── Element support sets (imported by per-category scripts) ──────────────────
_ORGANIC_Z = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})                  # H C N O F P S Cl Br I
_AIMNET2_Z = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53})   # + B Si As Se
_FENNIX_Z  = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53})           # + B Si
_ALL_Z     = frozenset(range(1, 90))

RESULT_FIELDS = [
    "system", "n_atoms", "model", "dtype", "device",
    "n_warmup", "n_steps", "elapsed_s", "steps_per_s", "ms_per_step",
    "vram_mib", "status", "error",
]
MAE_RESULT_FIELDS = [
    "model", "dtype", "device", "n_molecules", "mae_kj_mol", "mae_kcal_mol", "mae_ev",
    "status", "error",
]
MAE_PER_MOL_FIELDS = ["name", "n_atoms", "charge", "mae_kj_mol"]

_HARTREE_TO_EV  = 27.211386245988
_BOHR_TO_ANG    = 0.529177210903
_HARTREE_TO_KJ  = 2625.4996394798254
_EV_TO_KJ       = 96.48533212331002

# ── PDB utilities ─────────────────────────────────────────────────────────────
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
    syms, pos = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                syms.append(_sym(line[12:16]))
                pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return Atoms(symbols=syms, positions=pos, pbc=False)

def load_atoms(sys_def: dict) -> Atoms:
    pdb = Path(sys_def["pdb"])
    if _has_element_col(pdb):
        atoms = ase_read(str(pdb))
        atoms.pbc = False
    else:
        atoms = read_pdb_vacuum(pdb)
    atoms.info["charge"]    = sys_def["charge"]
    atoms.info["spin"]      = sys_def["spin"]
    atoms.info["_pdb_path"] = str(pdb)
    return atoms


# ── Workers (spawned subprocesses) ────────────────────────────────────────────
# Both workers import __main__ to call the per-category create_calculator and
# the optional configure_atoms hook.

def _worker(queue, system_name, model_name, dtype, device, n_warmup, n_steps):
    try:
        import __main__
        sys_def = next(s for s in SYSTEMS if s["name"] == system_name)

        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        atoms = load_atoms(sys_def)

        mem_before = (torch.cuda.memory_reserved(torch.device(device))
                      if device.startswith("cuda") else 0)

        calc  = __main__.create_calculator(model_name, device, dtype)
        atoms.calc = calc

        configure = getattr(__main__, "configure_atoms", None)
        if configure:
            configure(atoms, calc, model_name, sys_def["charge"], sys_def["spin"])

        # L-BFGS minimization to remove large forces
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=0.1, steps=50)

        # 100-step equilibration with trajectory
        MaxwellBoltzmannDistribution(atoms, temperature_K=300.0,
                                     rng=np.random.default_rng(42))
        atoms.set_constraint(constraints.FixCom())
        dyn = Langevin(atoms, timestep=units.fs, temperature_K=300.0,
                       friction=0.001 / units.fs, fixcm=False)

        device_str = device.replace(":", "")
        safe_name  = re.sub(r"[^\w.-]", "_", model_name)
        traj_path  = PROJECT / "trajectories" / f"{system_name}_{safe_name}_{dtype}_{device_str}.xyz"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        dyn.attach(lambda: write(str(traj_path), atoms, append=True), interval=10)
        dyn.run(100)

        dyn.observers.clear()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dyn.run(n_steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        mem_after = (torch.cuda.memory_reserved(torch.device(device))
                     if device.startswith("cuda") else 0)
        vram_mib  = round((mem_after - mem_before) / 1024**2, 1)

        queue.put({
            "system":      system_name,
            "n_atoms":     len(atoms),
            "elapsed_s":   round(elapsed, 4),
            "steps_per_s": round(n_steps / elapsed, 2),
            "ms_per_step": round(1000 * elapsed / n_steps, 2),
            "vram_mib":    vram_mib,
            "status":      "ok",
            "error":       "",
        })
    except Exception:
        queue.put({"status": "error", "error": traceback.format_exc().splitlines()[-1]})


def _mae_worker(queue, model_name, device, dtype, hdf5_path):
    try:
        import h5py
        import __main__

        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        calc      = __main__.create_calculator(model_name, device, dtype)
        configure = getattr(__main__, "configure_atoms", None)

        import __main__
        supported = getattr(__main__, "supported_elements", None)
        supported_z = supported(model_name) if supported else None

        mol_names   = []
        mol_sizes   = []
        mol_charges = []
        mol_maes_kj = []  # kJ/mol per molecule

        with h5py.File(hdf5_path, "r") as f:
            for mol_name, mol_data in f.items():
                atomic_numbers = mol_data["atomic_numbers"][:]

                if supported_z is not None and not frozenset(atomic_numbers).issubset(supported_z):
                    continue

                # SPICE stores conformations in Bohr; ASE needs Angstrom
                conformations = mol_data["conformations"][:] * _BOHR_TO_ANG
                # Use formation_energy (atomic references subtracted) in kJ/mol
                dft_energies  = mol_data["formation_energy"][:] * _HARTREE_TO_KJ

                charge, spin = 0, 1
                mol_smiles = None
                if "smiles" in mol_data:
                    smiles = mol_data["smiles"].asstr()[0]
                    from openff.toolkit.topology import Molecule
                    mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
                    charge = int(mol.total_charge.m)
                    mol_smiles = smiles
                n_electrons = int(np.sum(atomic_numbers)) - charge
                spin = 1 if n_electrons % 2 == 0 else 2

                n_conf         = len(conformations)
                model_energies = np.empty(n_conf)

                for k, conf in enumerate(conformations):
                    atoms = Atoms(numbers=atomic_numbers, positions=conf, pbc=False)
                    atoms.calc = calc
                    if configure:
                        configure(atoms, calc, model_name, charge, spin)
                    model_energies[k] = atoms.get_potential_energy() * _EV_TO_KJ

                i_idx, j_idx = np.triu_indices(n_conf, k=1)
                mol_mae = float(np.mean(np.abs(
                    (model_energies[i_idx] - model_energies[j_idx]) -
                    (dft_energies[i_idx]   - dft_energies[j_idx])
                )))
                mol_names.append(mol_name)
                mol_sizes.append(int(len(atomic_numbers)))
                mol_charges.append(int(charge))
                mol_maes_kj.append(mol_mae)

        mae_kj = float(np.mean(mol_maes_kj))
        queue.put({
            "status":       "ok",
            "n_molecules":  len(mol_maes_kj),
            "mae_kj_mol":   round(mae_kj, 4),
            "mae_kcal_mol": round(mae_kj / 4.184, 4),
            "mae_ev":       round(mae_kj / _EV_TO_KJ, 6),
            "mol_names":    mol_names,
            "mol_sizes":    mol_sizes,
            "mol_charges":  mol_charges,
            "mol_maes_kj":  mol_maes_kj,
        })
    except Exception:
        queue.put({"status": "error", "error": traceback.format_exc().splitlines()[-1]})


def run_isolated(system_name, model_name, dtype, device, n_warmup, n_steps, timeout) -> dict:
    ctx   = mp.get_context("spawn")
    queue = ctx.Queue()
    proc  = ctx.Process(target=_worker,
                        args=(queue, system_name, model_name, dtype, device, n_warmup, n_steps))
    proc.start(); proc.join(timeout)
    if proc.is_alive():
        proc.kill(); proc.join()
        return {"status": "timeout", "error": f">{timeout}s"}
    if proc.exitcode == -9:
        return {"status": "killed", "error": "OOM/SIGKILL"}
    return queue.get_nowait() if not queue.empty() else {"status": "error", "error": f"exit {proc.exitcode}"}


def run_mae_isolated(model_name, device, dtype, timeout) -> dict:
    ctx   = mp.get_context("spawn")
    queue = ctx.Queue()
    proc  = ctx.Process(target=_mae_worker,
                        args=(queue, model_name, device, dtype, str(SPICE_HDF5)))
    proc.start(); proc.join(timeout)
    if proc.is_alive():
        proc.kill(); proc.join()
        return {"status": "timeout", "error": f">{timeout}s"}
    if proc.exitcode == -9:
        return {"status": "killed", "error": "OOM/SIGKILL"}
    return queue.get_nowait() if not queue.empty() else {"status": "error", "error": f"exit {proc.exitcode}"}


# ── System info ───────────────────────────────────────────────────────────────
def collect_system_info() -> dict:
    info: dict = {}
    info["python"]     = sys.version.split()[0]
    info["torch"]      = torch.__version__
    info["torch_cuda"] = torch.version.cuda or "N/A"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"]               = props.name
        info["gpu_vram_gib"]           = round(props.total_memory / 1024**3, 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_sm_count"]           = props.multi_processor_count
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,clocks.max.memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
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
    info["cpu_logical_cores"] = os.cpu_count()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if "model name" in line:
                info["cpu_model"] = line.split(":", 1)[1].strip(); break
        pairs = set(zip(re.findall(r"physical id\s*:\s*(\d+)", cpuinfo),
                        re.findall(r"core id\s*:\s*(\d+)", cpuinfo)))
        if pairs:
            info["cpu_physical_cores"] = len(pairs)
    except Exception:
        pass
    try:
        freq = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read_text()
        info["cpu_max_freq_mhz"] = round(int(freq.strip()) / 1000, 1)
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["ram_total_gib"] = round(int(line.split()[1]) / 1024**2, 2)
            elif line.startswith("MemAvailable:"):
                info["ram_available_gib"] = round(int(line.split()[1]) / 1024**2, 2)
    except Exception:
        pass
    return info


# ── Main loop (called from each per-category script) ─────────────────────────
def run_benchmark(all_models: list, supported_elements_fn, default_out: str):
    """
    all_models: list of model names this script handles
    supported_elements_fn: model_name -> frozenset[int]
    default_out: default CSV filename (relative to PROJECT)
    """
    p = argparse.ArgumentParser()
    p.add_argument("--devices",       nargs="+", default=None)
    p.add_argument("--models",        nargs="+", default=all_models)
    p.add_argument("--dtypes",        nargs="+", default=["float32"])
    p.add_argument("--systems",       nargs="+", default=None)
    p.add_argument("--n-warmup",      type=int,   default=100)
    p.add_argument("--n-steps",       type=int,   default=100)
    p.add_argument("--max-step-time", type=float, default=10.0)
    p.add_argument("--out",           default=default_out)
    p.add_argument("--spice-mae",      action="store_true")
    p.add_argument("--spice-mae-only", action="store_true",
                   help="Run SPICE MAE only; skip MD entirely.")
    p.add_argument("--mae-timeout",   type=int,   default=7200)
    args = p.parse_args()
    if args.spice_mae_only:
        args.spice_mae = True

    unknown = [m for m in args.models if m not in all_models]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Available: {all_models}")

    devices = args.devices or (
        ["cuda:0"] if torch.cuda.is_available() else
        (_ for _ in ()).throw(SystemExit("No CUDA device. Pass --devices cpu."))
    )
    if not args.devices and not torch.cuda.is_available():
        raise SystemExit("No CUDA device found. Pass --devices cpu to run on CPU.")
    if not args.devices:
        devices = ["cuda:0"]

    systems = SYSTEMS
    if args.systems:
        systems = [s for s in SYSTEMS if s["name"] in set(args.systems)]
        if not systems:
            raise SystemExit(f"No matching systems. Available: {[s['name'] for s in SYSTEMS]}")

    for s in systems:
        if not Path(s["pdb"]).exists():
            raise SystemExit(f"Missing PDB: {s['pdb']}")

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.spice_mae_only:
        _loaded = {s["name"]: load_atoms(s) for s in systems}
        system_elements = {name: frozenset(a.get_atomic_numbers()) for name, a in _loaded.items()}
        system_sizes    = {name: len(a)                            for name, a in _loaded.items()}
        del _loaded
        timeout = int((args.n_warmup + args.n_steps) * args.max_step_time)

        total   = len(devices) * len(args.models) * len(args.dtypes) * len(systems)
        done    = 0
        results: list[dict] = []
        oom_threshold: dict[tuple, int] = {}

        def _is_oom(row: dict) -> bool:
            return row.get("status") == "killed" or "out of memory" in row.get("error", "").lower()

        for device in devices:
            for model_name in args.models:
                for dtype in args.dtypes:
                    for sys_def in systems:
                        done += 1
                        tag = (f"[{done}/{total}]  {sys_def['name']:<12}"
                               f"  {model_name:<20}  {dtype:<8}  {device}")
                        print(tag, end="  ", flush=True)

                        n_atoms = system_sizes[sys_def["name"]]
                        key     = (model_name, dtype, device)

                        unsupported = system_elements[sys_def["name"]] - supported_elements_fn(model_name)
                        if unsupported:
                            from ase.data import chemical_symbols
                            syms = ", ".join(chemical_symbols[z] for z in sorted(unsupported))
                            print(f"SKIP (unsupported elements: {syms})")
                            row = {"status": "skipped", "error": f"unsupported elements: {syms}"}
                        elif n_atoms > oom_threshold.get(key, float("inf")):
                            oom_n = oom_threshold[key]
                            print(f"SKIP (OOM on {oom_n}-atom system)")
                            row = {"status": "skipped", "error": f"OOM on {oom_n}-atom system"}
                        else:
                            row = run_isolated(sys_def["name"], model_name, dtype, device,
                                               args.n_warmup, args.n_steps, timeout)
                            if row.get("status") == "ok":
                                print(f"{row['steps_per_s']:>8.2f} steps/s"
                                      f"  ({row['ms_per_step']:.2f} ms/step)")
                            else:
                                print(f"{row['status'].upper()}: {row['error']}")
                                if _is_oom(row):
                                    prev = oom_threshold.get(key, float("inf"))
                                    oom_threshold[key] = min(prev, n_atoms)

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
                        row.setdefault("vram_mib",    "")
                        row.setdefault("error",       "")
                        results.append(row)

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(results)

        n_ok = sum(1 for r in results if r.get("status") == "ok")
        print(f"\nDone. {n_ok}/{len(results)} ok  →  {out_path}")

    sys_info_path = out_path.with_name("system_info.json")
    if not sys_info_path.exists():
        with open(sys_info_path, "w") as f:
            json.dump(collect_system_info(), f, indent=2)

    if args.spice_mae:
        if not SPICE_HDF5.exists():
            print(f"\nSPICE HDF5 not found at {SPICE_HDF5}, skipping MAE.")
            return
        total_mae = len(devices) * len(args.models) * len(args.dtypes)
        done_mae  = 0
        mae_results: list[dict] = []
        print(f"\n── SPICE energy MAE  ({total_mae} runs) ──")
        per_mol_dir = out_path.parent / "mae_per_mol"
        per_mol_dir.mkdir(parents=True, exist_ok=True)

        for device in devices:
            for model_name in args.models:
                for dtype in args.dtypes:
                    done_mae += 1
                    print(f"[{done_mae}/{total_mae}]  {model_name:<20}  {dtype}  {device}  ",
                          end="", flush=True)
                    row = run_mae_isolated(model_name, device, dtype, args.mae_timeout)
                    if row.get("status") == "ok":
                        print(f"MAE = {row['mae_kj_mol']:.4f} kJ/mol"
                              f"  ({row['mae_kcal_mol']:.4f} kcal/mol)"
                              f"  n={row['n_molecules']}")
                        safe_name = re.sub(r"[^\w.-]", "_", model_name)
                        per_mol_path = per_mol_dir / f"{safe_name}.csv"
                        with open(per_mol_path, "w", newline="") as pf:
                            writer = csv.DictWriter(pf, fieldnames=MAE_PER_MOL_FIELDS)
                            writer.writeheader()
                            for nm, sz, ch, err in zip(
                                row["mol_names"], row["mol_sizes"],
                                row["mol_charges"], row["mol_maes_kj"]
                            ):
                                writer.writerow({"name": nm, "n_atoms": sz,
                                                 "charge": ch, "mae_kj_mol": round(err, 6)})
                    else:
                        print(f"{row['status'].upper()}: {row.get('error', '')}")
                    row.setdefault("model",        model_name)
                    row.setdefault("dtype",        dtype)
                    row.setdefault("device",       device)
                    row.setdefault("n_molecules",  "")
                    row.setdefault("mae_kj_mol",   "")
                    row.setdefault("mae_kcal_mol", "")
                    row.setdefault("mae_ev",       "")
                    row.setdefault("error",        "")
                    # drop per-mol lists before writing summary row
                    for k in ("mol_names", "mol_sizes", "mol_charges", "mol_maes_kj"):
                        row.pop(k, None)
                    mae_results.append(row)

        mae_out = out_path.with_name(out_path.stem.replace("results", "mae") + ".csv")
        with open(mae_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MAE_RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(mae_results)
        n_ok = sum(1 for r in mae_results if r.get("status") == "ok")
        print(f"MAE done. {n_ok}/{len(mae_results)} ok  →  {mae_out}")
