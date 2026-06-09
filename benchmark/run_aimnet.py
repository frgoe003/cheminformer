#!/usr/bin/env python
"""AIMNet2 model.  Env: ase-aimnet"""
from benchmark_common import _AIMNET2_Z, run_benchmark


ALL_MODELS = ["AIMNet2"]


def supported_elements(model_name: str) -> frozenset:
    return _AIMNET2_Z


def create_calculator(model_name: str, device: str, dtype: str):
    try:
        from aimnet.calculators import AIMNet2ASE
        return AIMNet2ASE("aimnet2")
    except ImportError:
        from aimnet.calculators import AIMNet2Calculator
        return AIMNet2Calculator("aimnet2")


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int):
    calc.set_charge(charge)
    calc.set_mult(spin)


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_aimnet.csv")
