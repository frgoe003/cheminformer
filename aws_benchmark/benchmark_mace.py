#!/usr/bin/env python
"""
MACE-POLAR-1 throughput benchmark.

Sweeps system × model × dtype × device and reports steps/second.
Each combination runs in an isolated spawned process so an OOM kill
cannot take down the whole sweep.  Results are saved to benchmark/results.csv.

Usage:
    python benchmark_mace.py                          # all combinations
    python benchmark_mace.py --devices cpu            # CPU only
    python benchmark_mace.py --systems capped_ala chignolin --models polar-1-s
    python benchmark_mace.py --n-warmup 20 --n-steps 100 --timeout 300
"""

import argparse
import csv
import multiprocessing as mp
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

try:
    from aws_benchmark.run_md_mace_polar import read_pdb_vacuum, _has_element_col
except ModuleNotFoundError:
    from run_md_mace_polar import read_pdb_vacuum, _has_element_col

PROJECT     = Path(__file__).resolve().parent
SYSTEMS_DIR = PROJECT / "mols"

SYSTEMS = [
    dict(name="capped_ala", pdb=SYSTEMS_DIR / "capped_ala.pdb",  charge=0, spin=1),
    dict(name="chignolin",  pdb=SYSTEMS_DIR / "chignolin.pdb",   charge=0, spin=1),
    dict(name="ubiquitin",  pdb=SYSTEMS_DIR / "ubiquitin.pdb",   charge=0, spin=1),
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


def _worker(queue, system_name, model_name, dtype, device, n_warmup, n_steps):
    """Runs in a spawned child process; puts a result dict into queue."""
    try:
        sys_def = next(s for s in SYSTEMS if s["name"] == system_name)

        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        atoms = load_atoms(sys_def)
        calc  = mace_polar(model=model_name, device=device,
                           default_dtype=dtype, enable_cueq=False)
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


def main():
    p = argparse.ArgumentParser(description="MACE-POLAR-1 throughput benchmark")
    p.add_argument("--devices",  nargs="+", default=None,
                   help="Devices to test (default: cuda:0 if available + cpu)")
    p.add_argument("--models",   nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--dtypes",   nargs="+", default=ALL_DTYPES, choices=ALL_DTYPES)
    p.add_argument("--systems",  nargs="+", default=None,
                   help="System names: capped_ala chignolin ubiquitin")
    p.add_argument("--n-warmup", type=int, default=10,
                   help="Warmup steps discarded before timing (default: 10)")
    p.add_argument("--n-steps",  type=int, default=50,
                   help="Steps timed per run (default: 50)")
    p.add_argument("--timeout",  type=int, default=600,
                   help="Per-run timeout in seconds (default: 600)")
    p.add_argument("--out",      default="benchmark/results.csv")
    args = p.parse_args()

    if args.devices:
        devices = args.devices
    else:
        devices = []
        if torch.cuda.is_available():
            devices.append("cuda:0")
        devices.append("cpu")

    systems = SYSTEMS
    if args.systems:
        names = set(args.systems)
        systems = [s for s in SYSTEMS if s["name"] in names]
        if not systems:
            raise SystemExit(f"No matching systems. Available: {[s['name'] for s in SYSTEMS]}")

    for s in systems:
        if not Path(s["pdb"]).exists():
            raise SystemExit(f"Missing: {s['pdb']}")

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
                           f"  {model_name:<12}  {dtype:<8}  {device}")
                    print(tag, end="  ", flush=True)

                    row = run_isolated(
                        sys_def["name"], model_name, dtype, device,
                        args.n_warmup, args.n_steps, args.timeout,
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


if __name__ == "__main__":
    main()
