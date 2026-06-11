#!/usr/bin/env python
"""
OpenMM-ML speed benchmark — all available MLPotential models.

Uses OpenMM as the MD engine (LangevinMiddleIntegrator) rather than ASE,
which is the key difference from the other per-category scripts.
Each model/system pair runs in an isolated spawned subprocess.

Environment: openmm-ml  (see setup.sh for the full install)

Usage:
    python run_openmm_ml.py
    python run_openmm_ml.py --models ani2x mace-off23-small aceff-1.1
    python run_openmm_ml.py --systems capped_ala chignolin --n-steps 50
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT     = Path(__file__).resolve().parent
SYSTEMS_DIR = PROJECT / "systems"

SYSTEMS = [
    dict(name="capped_ala", pdb=SYSTEMS_DIR / "capped_ala_22.pdb",  charge=0, spin=1),
    dict(name="chignolin",  pdb=SYSTEMS_DIR / "chignolin_138.pdb",  charge=0, spin=1),
    dict(name="ubiquitin",  pdb=SYSTEMS_DIR / "ubiquitin_602.pdb",  charge=0, spin=1),
    dict(name="2LZM",       pdb=SYSTEMS_DIR / "2LZM.pdb",           charge=0, spin=1),
    dict(name="1ZG4",       pdb=SYSTEMS_DIR / "1ZG4_2p2k.pdb",      charge=0, spin=1),
    dict(name="3N5G",       pdb=SYSTEMS_DIR / "3N5G.pdb",           charge=0, spin=1),
    dict(name="5G1P",       pdb=SYSTEMS_DIR / "5G1P_13k.pdb",       charge=0, spin=1),
    dict(name="1B3B",       pdb=SYSTEMS_DIR / "1B3B_19k.pdb",       charge=0, spin=1),
    dict(name="9VM6",       pdb=SYSTEMS_DIR / "9VM6.pdb",           charge=0, spin=1),
    dict(name="water_99k",  pdb=SYSTEMS_DIR / "water_99k.pdb",      charge=0, spin=1),
]

# ── Element support ───────────────────────────────────────────────────────────
_ANI1CCX_Z = frozenset({1, 6, 7, 8})
_ANI2X_Z   = frozenset({1, 6, 7, 8, 9, 16, 17})
_ORGANIC_Z = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
_AIMNET2_Z = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53})
_FENNIX_Z  = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53})
_ALL_Z     = frozenset(range(1, 90))

# ── Models ────────────────────────────────────────────────────────────────────
_ANI_MODELS    = ["ani1ccx", "ani2x"]
_MACE_MODELS   = [
    "mace-off23-small", "mace-off23-large", "mace-off24-medium",
    "mace-mpa-0-medium", "mace-omat-0-medium", "mace-omol-0-extra-large",
]
_ACEFF_MODELS  = ["aceff-1.1", "aceff-2.0"]
_AIMNET_MODELS = ["aimnet2"]
_ORB_MODELS    = ["orb-v3-conservative-omol", "orb-v3-conservative-inf-omat"]
_FENNIX_MODELS = ["fennix-bio1-small", "fennix-bio1-medium"]

ALL_MODELS = (
    _ANI_MODELS + _MACE_MODELS + _ACEFF_MODELS +
    _AIMNET_MODELS + _ORB_MODELS + _FENNIX_MODELS
)

RESULT_FIELDS = [
    "system", "n_atoms", "model", "dtype", "device",
    "n_warmup", "n_steps", "elapsed_s", "steps_per_s", "ms_per_step",
    "vram_mib", "avg_power_w", "peak_power_w", "status", "error",
]


def supported_elements(model_name: str) -> frozenset:
    if model_name == "ani1ccx":
        return _ANI1CCX_Z
    if model_name == "ani2x":
        return _ANI2X_Z
    if model_name in {"mace-off23-small", "mace-off23-large", "mace-off24-medium"}:
        return _ORGANIC_Z
    if model_name in _ACEFF_MODELS:
        return _ORGANIC_Z
    if model_name in _AIMNET_MODELS:
        return _AIMNET2_Z
    if model_name in _FENNIX_MODELS:
        return _FENNIX_Z
    # mace-mpa-0, mace-omat-0, mace-omol-0, orb: broad periodic-table coverage
    return _ALL_Z


def _model_create_system_kwargs(model_name: str, charge: int, spin: int) -> dict:
    """Extra kwargs forwarded to MLPotential.createSystem()."""
    kwargs: dict = {}
    # MACE foundation models: single precision is recommended for MD
    if model_name in {
        "mace-off23-small", "mace-off23-large", "mace-off24-medium",
        "mace-mpa-0-medium", "mace-omat-0-medium", "mace-omol-0-extra-large",
    }:
        kwargs["precision"] = "single"
    # Models that accept a total charge
    if model_name in _ACEFF_MODELS:
        kwargs["charge"] = charge
    if model_name in _AIMNET_MODELS:
        kwargs["charge"] = charge
    if model_name in _FENNIX_MODELS:
        kwargs["charge"] = charge
    if model_name in _ORB_MODELS:
        kwargs["charge"] = charge
        kwargs["spin"]   = spin
    return kwargs


# ── nvidia-smi helpers ────────────────────────────────────────────────────────

def _vram_mib(device_index: int) -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


def _sample_gpu_power(stop_event: threading.Event, samples: list,
                      device_index: int = 0, interval: float = 0.2) -> None:
    while not stop_event.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--id={device_index}",
                 "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                samples.append(float(r.stdout.strip()))
        except Exception:
            pass
        stop_event.wait(interval)


# ── Worker (runs in isolated subprocess) ──────────────────────────────────────

def _worker(queue, system_name, model_name, device, n_warmup, n_steps):
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from openmmml import MLPotential

        sys_def    = next(s for s in SYSTEMS if s["name"] == system_name)
        pdb        = app.PDBFile(str(sys_def["pdb"]))
        topology   = pdb.topology
        positions  = pdb.positions
        n_atoms    = sum(1 for _ in topology.atoms())

        gpu_idx     = int(device.split(":")[-1]) if ":" in device else 0
        vram_before = _vram_mib(gpu_idx) if device.startswith("cuda") else 0.0

        kwargs    = _model_create_system_kwargs(model_name, sys_def["charge"], sys_def["spin"])
        potential = MLPotential(model_name)
        system    = potential.createSystem(topology, **kwargs)

        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds,
        )
        if device.startswith("cuda"):
            idx      = device.split(":")[-1] if ":" in device else "0"
            platform = openmm.Platform.getPlatformByName("CUDA")
            props    = {"DeviceIndex": idx, "Precision": "mixed"}
        else:
            platform = openmm.Platform.getPlatformByName("CPU")
            props    = {}

        sim = app.Simulation(topology, system, integrator, platform, props)
        sim.context.setPositions(positions)
        sim.minimizeEnergy(maxIterations=50)

        # Warmup (equilibrate, not timed)
        sim.step(n_warmup)

        vram_after = _vram_mib(gpu_idx) if device.startswith("cuda") else 0.0
        vram_mib   = round(vram_after - vram_before, 1)

        # ── Timed run ─────────────────────────────────────────────────────────
        power_samples: list = []
        power_stop    = threading.Event()
        if device.startswith("cuda"):
            power_thread = threading.Thread(
                target=_sample_gpu_power,
                args=(power_stop, power_samples, gpu_idx, 0.2),
                daemon=True,
            )
            power_thread.start()
        else:
            power_thread = None

        t0 = time.perf_counter()
        sim.step(n_steps)
        elapsed = time.perf_counter() - t0

        if power_thread is not None:
            power_stop.set()
            power_thread.join(timeout=2.0)

        avg_power  = round(float(np.mean(power_samples)), 1) if power_samples else ""
        peak_power = round(float(np.max(power_samples)),  1) if power_samples else ""

        queue.put({
            "system":       system_name,
            "n_atoms":      n_atoms,
            "elapsed_s":    round(elapsed, 4),
            "steps_per_s":  round(n_steps / elapsed, 2),
            "ms_per_step":  round(1000 * elapsed / n_steps, 2),
            "vram_mib":     vram_mib,
            "avg_power_w":  avg_power,
            "peak_power_w": peak_power,
            "status":       "ok",
            "error":        "",
        })
    except Exception:
        queue.put({"status": "error", "error": traceback.format_exc().splitlines()[-1]})


def run_isolated(system_name, model_name, device, n_warmup, n_steps, timeout) -> dict:
    ctx   = mp.get_context("spawn")
    queue = ctx.Queue()
    proc  = ctx.Process(
        target=_worker,
        args=(queue, system_name, model_name, device, n_warmup, n_steps),
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill(); proc.join()
        return {"status": "timeout", "error": f">{timeout}s"}
    if proc.exitcode == -9:
        return {"status": "killed", "error": "OOM/SIGKILL"}
    return queue.get_nowait() if not queue.empty() else {"status": "error", "error": f"exit {proc.exitcode}"}


def collect_system_info() -> dict:
    try:
        sys.path.insert(0, str(PROJECT))
        from benchmark_common import collect_system_info as _csi
        return _csi()
    except Exception:
        return {}


def main():
    p = argparse.ArgumentParser(description="OpenMM-ML speed benchmark")
    p.add_argument("--devices",       nargs="+", default=None)
    p.add_argument("--models",        nargs="+", default=ALL_MODELS)
    p.add_argument("--systems",       nargs="+", default=None)
    p.add_argument("--n-warmup",      type=int,   default=100)
    p.add_argument("--n-steps",       type=int,   default=100)
    p.add_argument("--max-step-time", type=float, default=10.0,
                   help="Max wall-clock seconds per step for timeout (default: 10)")
    p.add_argument("--out",           default="results_openmm_ml.csv")
    args = p.parse_args()

    unknown = [m for m in args.models if m not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Available: {ALL_MODELS}")

    if args.devices:
        devices = args.devices
    else:
        try:
            from openmm import Platform
            cuda_available = any(
                Platform.getPlatform(i).getName() == "CUDA"
                for i in range(Platform.getNumPlatforms())
            )
        except Exception:
            cuda_available = False
        if not cuda_available:
            raise SystemExit("No CUDA platform found. Pass --devices cpu.")
        devices = ["cuda:0"]

    systems = SYSTEMS
    if args.systems:
        systems = [s for s in SYSTEMS if s["name"] in set(args.systems)]
        if not systems:
            raise SystemExit(f"No matching systems. Available: {[s['name'] for s in SYSTEMS]}")

    for s in systems:
        if not Path(s["pdb"]).exists():
            raise SystemExit(f"Missing PDB: {s['pdb']}")

    # Read element sets and atom counts up front (fast — just topology parsing)
    from openmm.app import PDBFile as _PDB
    system_elements: dict[str, frozenset] = {}
    system_sizes:    dict[str, int]       = {}
    for s in systems:
        pdb = _PDB(str(s["pdb"]))
        system_elements[s["name"]] = frozenset(
            a.element.atomic_number
            for a in pdb.topology.atoms() if a.element is not None
        )
        system_sizes[s["name"]] = sum(1 for _ in pdb.topology.atoms())

    timeout  = int((args.n_warmup + args.n_steps) * args.max_step_time)
    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total   = len(devices) * len(args.models) * len(systems)
    done    = 0
    results: list[dict] = []
    oom_threshold: dict[tuple, int] = {}

    def _is_oom(row: dict) -> bool:
        return row.get("status") == "killed" or "out of memory" in row.get("error", "").lower()

    for device in devices:
        for model_name in args.models:
            for sys_def in systems:
                done += 1
                tag = (f"[{done}/{total}]  {sys_def['name']:<12}"
                       f"  {model_name:<28}  {device}")
                print(tag, end="  ", flush=True)

                n_atoms = system_sizes[sys_def["name"]]
                key     = (model_name, device)

                unsupported = system_elements[sys_def["name"]] - supported_elements(model_name)
                if unsupported:
                    try:
                        from openmm.app.element import Element
                        syms = ", ".join(
                            Element.getByAtomicNumber(z).symbol for z in sorted(unsupported)
                        )
                    except Exception:
                        syms = str(sorted(unsupported))
                    print(f"SKIP (unsupported elements: {syms})")
                    row = {"status": "skipped", "error": f"unsupported elements: {syms}"}
                elif n_atoms > oom_threshold.get(key, float("inf")):
                    oom_n = oom_threshold[key]
                    print(f"SKIP (OOM on {oom_n}-atom system)")
                    row = {"status": "skipped", "error": f"OOM on {oom_n}-atom system"}
                else:
                    row = run_isolated(sys_def["name"], model_name, device,
                                       args.n_warmup, args.n_steps, timeout)
                    if row.get("status") == "ok":
                        print(f"{row['steps_per_s']:>8.2f} steps/s"
                              f"  ({row['ms_per_step']:.2f} ms/step)")
                    else:
                        print(f"{row['status'].upper()}: {row['error']}")
                        if _is_oom(row):
                            prev = oom_threshold.get(key, float("inf"))
                            oom_threshold[key] = min(prev, n_atoms)

                row.setdefault("system",       sys_def["name"])
                row.setdefault("n_atoms",      "?")
                row.setdefault("model",        model_name)
                row.setdefault("dtype",        "float32")
                row.setdefault("device",       device)
                row.setdefault("n_warmup",     args.n_warmup)
                row.setdefault("n_steps",      args.n_steps)
                row.setdefault("elapsed_s",    "")
                row.setdefault("steps_per_s",  "")
                row.setdefault("ms_per_step",  "")
                row.setdefault("vram_mib",     "")
                row.setdefault("avg_power_w",  "")
                row.setdefault("peak_power_w", "")
                row.setdefault("error",        "")
                results.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nDone. {n_ok}/{len(results)} ok  →  {out_path}")

    sys_info_path = out_path.parent / "system_info.json"
    if not sys_info_path.exists():
        info = collect_system_info()
        if info:
            with open(sys_info_path, "w") as f:
                json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
