#!/usr/bin/env python
"""
Vacuum MD with MACE-POLAR-1 for any PDB file.

Backbone dihedrals (phi, psi) of capped alanine are optionally collected and
plotted; enable with --track-dihedrals (only meaningful for capped_ala.pdb).
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms, constraints, units
from ase.io import read as ase_read
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import mace_polar

# ── PDB readers (imported by benchmark_mace.py) ───────────────────────────────

def _has_element_col(path: Path) -> bool:
    with open(path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                return len(line.rstrip('\n')) >= 78 and line[76:78].strip() != ''
    return False

def _sym(name: str) -> str:
    s = re.sub(r'^\d+', '', name.strip())
    for prefix in ('CA', 'CB', 'CH', 'C'):
        if s.startswith(prefix): return 'C'
    if s.startswith('N'): return 'N'
    if s.startswith('O'): return 'O'
    if s.startswith('H'): return 'H'
    return s[0].upper()

def read_pdb_vacuum(path: Path) -> Atoms:
    """Fallback reader for PDBs with non-standard atom names."""
    syms, pos = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                syms.append(_sym(line[12:16]))
                pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return Atoms(symbols=syms, positions=pos, pbc=False)

def read_structure(path: Path) -> Atoms:
    if _has_element_col(path):
        atoms = ase_read(str(path))
        atoms.pbc = False
    else:
        atoms = read_pdb_vacuum(path)
    return atoms

# ── Dihedral helpers (capped alanine only) ────────────────────────────────────
# phi: C(ACE)–N(ALA)–CA(ALA)–C(ALA)  → [4, 6, 8, 14]
# psi: N(ALA)–CA(ALA)–C(ALA)–N(NME) → [6, 8, 14, 16]
PHI_IDX = [4, 6, 8, 14]
PSI_IDX = [6, 8, 14, 16]

def dihedral_angle(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2); n1 /= np.linalg.norm(n1)
    n2 = np.cross(b2, b3); n2 /= np.linalg.norm(n2)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

def plot_fes(phi, psi, temperature_K, out_path, bins=18):
    kBT   = units.kB * temperature_K
    edges = np.linspace(-180, 180, bins + 1)
    H, _, _ = np.histogram2d(phi, psi, bins=edges, density=True)
    with np.errstate(divide='ignore'):
        fes = np.where(H > 0, -kBT * np.log(H), np.nan)
    fes -= np.nanmin(fes)
    cx = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sc = axes[0].scatter(phi, psi, c=np.arange(len(phi)), cmap="plasma", s=30, zorder=3)
    axes[0].set(xlim=(-180,180), ylim=(-180,180), xlabel="φ (°)", ylabel="ψ (°)",
                title="Ramachandran trajectory")
    axes[0].axhline(0, lw=0.5, c="gray"); axes[0].axvline(0, lw=0.5, c="gray")
    plt.colorbar(sc, ax=axes[0], label="MD step")

    cf = axes[1].contourf(cx, cx, fes.T, levels=15, cmap="RdYlBu_r")
    axes[1].contour(cx, cx, fes.T, levels=15, colors="k", linewidths=0.3, alpha=0.5)
    axes[1].set(xlim=(-180,180), ylim=(-180,180), xlabel="φ (°)", ylabel="ψ (°)",
                title=f"2-D FES  (−k_BT ln P,  T={temperature_K} K)")
    axes[1].axhline(0, lw=0.5, c="gray"); axes[1].axvline(0, lw=0.5, c="gray")
    plt.colorbar(cf, ax=axes[1], label="FES (eV)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    def _parse_args():
        p = argparse.ArgumentParser(description="Vacuum MD with MACE-POLAR-1")
        p.add_argument("--pdb", default=None,
                       help="Path to input PDB (default: mols/capped_ala.pdb next to this script)")
        p.add_argument("--device", default="cuda:0",
                       help="PyTorch device string: 'cpu', 'cuda', 'cuda:0', … (default: cuda:0)")
        p.add_argument("--model", default="polar-1-s",
                       choices=["polar-1-s", "polar-1-m", "polar-1-l"],
                       help="MACE-POLAR-1 checkpoint size (default: polar-1-s)")
        p.add_argument("--steps", type=int, default=50,
                       help="Number of MD steps (default: 50)")
        p.add_argument("--temperature", type=float, default=300.0,
                       help="Langevin temperature in K (default: 300)")
        p.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                       help="Floating-point precision (default: float32)")
        p.add_argument("--charge", type=int, default=0,
                       help="Net charge of the system (default: 0)")
        p.add_argument("--spin", type=int, default=1,
                       help="Spin multiplicity (default: 1)")
        p.add_argument("--write-every", type=int, default=10, metavar="N",
                       help="Write trajectory frame every N steps (default: 10)")
        p.add_argument("--log-every", type=int, default=1, metavar="N",
                       help="Print energy every N steps (default: 1)")
        p.add_argument("--track-dihedrals", action="store_true",
                       help="Track phi/psi dihedrals and plot FES (capped_ala only)")
        p.add_argument("--no-cueq", action="store_true",
                       help="Disable cuEquivariance acceleration")
        return p.parse_args()

    args = _parse_args()

    PROJECT = Path(__file__).resolve().parent
    PDB = PROJECT / "mols" / "capped_ala.pdb" if args.pdb is None else Path(args.pdb).resolve()

    if not PDB.exists():
        raise SystemExit(f"PDB not found: {PDB}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available — use --device cpu")
    if args.device == "cuda":
        args.device = "cuda:0"
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    OUT = PROJECT / "output" / PDB.stem
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        import cuequivariance_torch as _cueq_ops
        _ops_available = hasattr(_cueq_ops, "segmented_polynomial")
    except ImportError:
        _ops_available = False
    ENABLE_CUEQ = not args.no_cueq and _ops_available

    atoms = read_structure(PDB)
    atoms.info["charge"]         = args.charge
    atoms.info["spin"]           = args.spin
    atoms.info["external_field"] = [0.0, 0.0, 0.0]

    calc = mace_polar(model=args.model, device=args.device,
                      default_dtype=args.dtype, enable_cueq=ENABLE_CUEQ)
    atoms.calc = calc

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature,
                                 rng=np.random.default_rng(42))
    atoms.set_constraint(constraints.FixCom())

    dyn = Langevin(atoms, timestep=1 * units.fs, temperature_K=args.temperature,
                   friction=0.01 / units.fs, fixcm=False)

    phi_traj: list[float] = []
    psi_traj: list[float] = []
    traj_file = str(OUT / f"{PDB.stem}.xyz")
    _step_counter = [0]

    def _collect():
        _step_counter[0] += 1
        step = _step_counter[0]
        if args.track_dihedrals:
            pos = atoms.get_positions()
            phi_traj.append(dihedral_angle(*[pos[i] for i in PHI_IDX]))
            psi_traj.append(dihedral_angle(*[pos[i] for i in PSI_IDX]))
        if step % args.log_every == 0:
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            T    = ekin / (1.5 * len(atoms) * units.kB)
            dih  = (f"  φ={phi_traj[-1]:7.2f}°  ψ={psi_traj[-1]:7.2f}°"
                    if args.track_dihedrals else "")
            print(f"  step {step:6d}  Epot={epot:.4f} eV  Ekin={ekin:.4f} eV  T={T:.1f} K{dih}")

    dyn.attach(_collect, interval=1)
    dyn.attach(lambda: write(traj_file, atoms, append=True), interval=args.write_every)

    if args.device.startswith("cuda"):
        idx = torch.device(args.device).index or 0
        print(f"Device: {args.device}  ({torch.cuda.get_device_name(idx)})")
    else:
        print(f"Device: {args.device}")
    print(f"System: {PDB.name}  ({len(atoms)} atoms)")
    print(f"Model : {args.model}  (CuEq={'on' if ENABLE_CUEQ else 'off'}, dtype={args.dtype})")
    print(f"Steps : {args.steps}  |  write every {args.write_every}  |  log every {args.log_every}\n")

    dyn.run(args.steps)

    if args.track_dihedrals and phi_traj:
        phi_arr = np.array(phi_traj)
        psi_arr = np.array(psi_traj)
        np.savetxt(OUT / "dihedrals.csv",
                   np.column_stack([phi_arr, psi_arr]),
                   header="phi_deg,psi_deg", delimiter=",", comments="")
        plot_fes(phi_arr, psi_arr, args.temperature, out_path=OUT / "dihedral_fes.png")

    print("Trajectory saved →", traj_file)
