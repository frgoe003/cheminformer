#!/usr/bin/env python
"""GFN2-xTB via tblite ASE calculator. CPU-only. Env: ase-gfn2xtb"""

import sys

from benchmark_common import run_benchmark

# GFN2-xTB is parametrized for H–Rn (Z = 1–86)
_GFN2_Z = frozenset(range(1, 87))

ALL_MODELS = ["GFN2-xTB"]


def supported_elements(model_name: str) -> frozenset:
    return _GFN2_Z


def create_calculator(model_name: str, device: str, dtype: str):
    from tblite.ase import TBLite
    return TBLite(method="GFN2-xTB")


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int) -> None:
    calc.set(charge=charge, multiplicity=spin)
    atoms.info["charge"] = charge
    atoms.info["spin"]   = spin


if __name__ == "__main__":
    if "--devices" not in sys.argv:
        sys.argv += ["--devices", "cpu"]
    run_benchmark(ALL_MODELS, supported_elements, "results_gfn2xtb.csv")
