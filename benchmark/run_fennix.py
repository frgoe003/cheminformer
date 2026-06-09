#!/usr/bin/env python
"""FeNNix-Bio1 S/M models.  Env: ase-fennix"""
from benchmark_common import _FENNIX_Z, run_benchmark, PROJECT

ALL_MODELS = ["FeNNix-Bio1(S)", "FeNNix-Bio1(M)"]

_FILES = {
    "FeNNix-Bio1(S)": str(PROJECT / "fennix-bio1S.fnx"),
    "FeNNix-Bio1(M)": str(PROJECT / "fennix-bio1M.fnx"),
}


def supported_elements(model_name: str) -> frozenset:
    return _FENNIX_Z


def create_calculator(model_name: str, device: str, dtype: str):
    from fennol.ase import FENNIXCalculator
    return FENNIXCalculator(model=_FILES[model_name],
                            matmul_prec="highest", gpu_preprocessing=True)


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int):
    charges = [0] * len(atoms)
    charges[0] = charge
    atoms.set_initial_charges(charges)


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_fennix.csv")
