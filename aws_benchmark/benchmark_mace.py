#!/usr/bin/env python
"""
MACE-POLAR-1 throughput benchmark.

Sweeps system × model × dtype × device and reports steps/second.
Results are saved to benchmark/results.csv.

Usage:
    python benchmark_mace.py                          # all combinations
    python benchmark_mace.py --devices cpu            # CPU only
    python benchmark_mace.py --systems capped_ala chignolin --models polar-1-s
    python benchmark_mace.py --n-warmup 20 --n-steps 100
"""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from ase import constraints, units
from ase.io import read as ase_read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import mace_polar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_md_mace_polar import read_pdb_vacuum, _has_element_col

PROJECT     = Path(__file__).resolve().parent
SYSTEMS_DIR = PROJECT / "benchmark" / "systems"

# charge/spin are approximations — only throughput matters here
SYSTEMS = [
    dict(name="capped_ala", pdb=SYSTEMS_DIR / "capped_ala.pdb",  charge=0, spin=1),
    dict(name="chignolin",  pdb=SYSTEMS_DIR / "chignolin.pdb",   charge=0, spin=1),
    dict(name="ubiquitin",  pdb=SYSTEMS_DIR / "ubiquitin_H.pdb", charge=0, spin=1),
]

ALL_MODELS = ["polar-1-s", "polar-1-m", "polar-1-l"]
ALL_DTYPES = ["float32", "float64"]

RESULT_FIELDS = [
    "system", "n_atoms", "model", "dtype", "device",
    "n_warmup", "n_steps", "elapsed_s", "steps_per_s", "ms_per_step",
    "status", "error",
]


def load_atoms(sys_def: dict):
    pdb = Path(sys_def["pdb"])
    if _has_element_col(pdb):
        atoms = ase_read(str(pdb))
        atoms.pbc = False
    else:
        atoms = read_pdb_vacuum(pdb)
    atoms.info["charge"]         = sys_def["charge"]
    atoms.info["spin"]           = sys_def["spin"]
    atoms.info["external_field"] = [0.0, 0.0, 0.0]
    return atoms


def run_one(sys_def: dict, calc, device: str, n_warmup: int, n_steps: int) -> dict:
    atoms = load_atoms(sys_def)
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0,
                                 rng=np.random.default_rng(42))
    atoms.set_constraint(constraints.FixCom())

    dyn = Langevin(atoms, timestep=units.fs, temperature_K=300.0,
                   friction=0.01 / units.fs, fixcm=False)

    dyn.run(n_warmup)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    dyn.run(n_steps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "system":      sys_def["name"],
        "n_atoms":     len(atoms),
        "elapsed_s":   round(elapsed, 4),
        "steps_per_s": round(n_steps / elapsed, 2),
        "ms_per_step": round(1000 * elapsed / n_steps, 2),
        "status":      "ok",
        "error":       "",
    }


def main():
    p = argparse.ArgumentParser(description="MACE-POLAR-1 throughput benchmark")
    p.add_argument("--devices", nargs="+", default=None,
                   help="Devices to test (default: cuda:0 if available + cpu)")
    p.add_argument("--models",  nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS)
    p.add_argument("--dtypes",  nargs="+", default=ALL_DTYPES,
                   choices=ALL_DTYPES)
    p.add_argument("--systems", nargs="+", default=None,
                   help="System names: capped_ala chignolin ubiquitin")
    p.add_argument("--n-warmup", type=int, default=10,
                   help="Steps discarded for JIT/cuDNN warmup (default: 10)")
    p.add_argument("--n-steps",  type=int, default=50,
                   help="Steps timed per run (default: 50)")
    p.add_argument("--out", default="benchmark/results.csv",
                   help="Output CSV path (default: benchmark/results.csv)")
    args = p.parse_args()

    # Resolve devices
    if args.devices:
        devices = args.devices
    else:
        devices = []
        if torch.cuda.is_available():
            devices.append("cuda:0")
        devices.append("cpu")

    # Filter systems
    systems = SYSTEMS
    if args.systems:
        names = set(args.systems)
        systems = [s for s in SYSTEMS if s["name"] in names]
        if not systems:
            raise SystemExit(f"No matching systems. Available: {[s['name'] for s in SYSTEMS]}")

    # Check all PDB files exist
    for s in systems:
        if not Path(s["pdb"]).exists():
            raise SystemExit(
                f"Missing: {s['pdb']}\n"
                "Run create_benchmark.sh first to prepare all system files."
            )

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(devices) * len(args.models) * len(args.dtypes) * len(systems)
    done  = 0
    results: list[dict] = []

    for device in devices:
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        for model_name in args.models:
            for dtype in args.dtypes:
                # One calculator per (device, model, dtype), reused across systems
                try:
                    calc = mace_polar(model=model_name, device=device,
                                      default_dtype=dtype, enable_cueq=False)
                except Exception as e:
                    err = traceback.format_exc().splitlines()[-1]
                    for sys_def in systems:
                        done += 1
                        print(f"[{done}/{total}] {sys_def['name']} / {model_name} / {dtype} / {device}"
                              f"  → FAILED (calculator): {err}")
                        results.append({
                            "system": sys_def["name"], "n_atoms": "?",
                            "model": model_name, "dtype": dtype, "device": device,
                            "n_warmup": args.n_warmup, "n_steps": args.n_steps,
                            "elapsed_s": "", "steps_per_s": "", "ms_per_step": "",
                            "status": "error", "error": err,
                        })
                    continue

                for sys_def in systems:
                    done += 1
                    tag = (f"[{done}/{total}]  {sys_def['name']:<12}"
                           f"  {model_name:<12}  {dtype:<8}  {device}")
                    print(tag, end="  ", flush=True)
                    try:
                        row = run_one(sys_def, calc, device,
                                      n_warmup=args.n_warmup, n_steps=args.n_steps)
                        print(f"{row['steps_per_s']:>8.2f} steps/s"
                              f"  ({row['ms_per_step']:.2f} ms/step)")
                        row.update(model=model_name, dtype=dtype, device=device,
                                   n_warmup=args.n_warmup, n_steps=args.n_steps)
                    except Exception:
                        err = traceback.format_exc().splitlines()[-1]
                        print(f"FAILED: {err}")
                        row = {
                            "system": sys_def["name"], "n_atoms": "?",
                            "model": model_name, "dtype": dtype, "device": device,
                            "n_warmup": args.n_warmup, "n_steps": args.n_steps,
                            "elapsed_s": "", "steps_per_s": "", "ms_per_step": "",
                            "status": "error", "error": err,
                        }
                    results.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    n_ok  = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone. {n_ok} ok, {n_err} errors.")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
