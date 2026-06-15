#!/usr/bin/env python
"""g-xTB via xtb binary (--gxtb flag). CPU-only. Env: ase-gxtb"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write as ase_write

from benchmark_common import _ALL_Z, run_benchmark

_HARTREE_TO_EV = 27.211386245988
_BOHR_TO_ANG   = 0.529177210903

ALL_MODELS = ["g-xTB"]


class GXTBCalculator(Calculator):
    """Wraps the g-xTB binary (xtb --gxtb) as an ASE calculator.

    By default each call uses a fresh temporary directory (safe for independent
    molecules, e.g. SPICE MAE).  Pass persistent=True to keep the working
    directory across calls so xtb can warm-start the SCC from the previous
    step's .xtbrestart file — meaningfully faster for MD trajectories where
    successive geometries are similar.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, charge=0, uhf=0, xtb_binary="xtb", persistent=False, **kwargs):
        super().__init__(**kwargs)
        self.charge      = charge
        self.uhf         = uhf
        self.xtb_binary  = xtb_binary
        self.persistent  = persistent
        self._workdir    = None   # set on first call when persistent=True

    def _get_workdir(self):
        if self.persistent:
            if self._workdir is None:
                self._workdir = Path(tempfile.mkdtemp(prefix="gxtb_"))
            return self._workdir, False  # False = caller must NOT delete it
        tmp = Path(tempfile.mkdtemp(prefix="gxtb_"))
        return tmp, True  # True = caller should delete after use

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        charge = int(atoms.info.get("charge", self.charge))
        spin   = int(atoms.info.get("spin",   self.uhf + 1))
        uhf    = spin - 1

        workdir, cleanup = self._get_workdir()
        try:
            xyz_path = workdir / "struc.xyz"
            ase_write(str(xyz_path), atoms)

            cmd = [self.xtb_binary, str(xyz_path),
                   "--gxtb", "--grad",
                   "--chrg", str(charge),
                   "--uhf",  str(uhf)]
            if not self.persistent:
                cmd.append("--norestart")

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=workdir,
            )

            energy_ha = None
            for line in result.stdout.splitlines():
                m = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", line)
                if m:
                    energy_ha = float(m.group(1))
                    break
            if energy_ha is None:
                stdout_tail = result.stdout[-500:].strip().replace("\n", " | ")
                stderr_tail = result.stderr[-200:].strip().replace("\n", " | ")
                raise RuntimeError(
                    f"xtb --gxtb failed (rc={result.returncode}): {stderr_tail or stdout_tail}"
                )

            grad_path = workdir / "gradient"
            forces = (_parse_gradient(grad_path, len(atoms))
                      if grad_path.exists()
                      else np.zeros((len(atoms), 3)))
        finally:
            if cleanup:
                shutil.rmtree(workdir, ignore_errors=True)

        self.results["energy"] = energy_ha * _HARTREE_TO_EV
        self.results["forces"] = forces

    def __del__(self):
        if self._workdir is not None and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)


def _parse_gradient(grad_path: Path, n_atoms: int) -> np.ndarray:
    """Read turbomole gradient file → forces in eV/Å (negated gradient)."""
    state  = "seek"   # seek → header → coords → grads
    n_seen = 0
    grads  = []

    for line in grad_path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if state == "seek":
            if s.startswith("$grad"):
                state = "header"
        elif state == "header":
            if s.lower().startswith("cycle"):
                state = "coords"
        elif state == "coords":
            if len(s.split()) == 4:  # 3 coords + element symbol
                n_seen += 1
                if n_seen >= n_atoms:
                    state = "grads"
        elif state == "grads":
            if s.startswith("$"):
                break
            grads.append(s)
            if len(grads) >= n_atoms:
                break

    if len(grads) != n_atoms:
        raise RuntimeError(f"Expected {n_atoms} gradient lines, got {len(grads)}")

    conv = _HARTREE_TO_EV / _BOHR_TO_ANG  # Eh/Bohr → eV/Å
    forces = []
    for g in grads:
        fx, fy, fz = [float(v.upper().replace("D", "E")) for v in g.split()]
        forces.append([-fx * conv, -fy * conv, -fz * conv])
    return np.array(forces)


def supported_elements(model_name: str) -> frozenset:
    return _ALL_Z


def create_calculator(model_name: str, device: str, dtype: str) -> GXTBCalculator:
    return GXTBCalculator(persistent=False)


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int) -> None:
    calc.charge = charge
    calc.uhf    = spin - 1
    atoms.info["charge"] = charge
    atoms.info["spin"]   = spin


if __name__ == "__main__":
    if "--devices" not in sys.argv:
        sys.argv += ["--devices", "cpu"]
    run_benchmark(ALL_MODELS, supported_elements, "results_gxtb.csv")
