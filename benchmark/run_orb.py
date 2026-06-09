#!/usr/bin/env python
"""Orb-v3-omol model.  Env: ase-orb"""
from benchmark_common import _ALL_Z, run_benchmark

ALL_MODELS = ["Orb-v3-omol"]


def supported_elements(model_name: str) -> frozenset:
    return _ALL_Z


def create_calculator(model_name: str, device: str, dtype: str):
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    orbff = pretrained.orb_v3_conservative_omol(device=device, precision=f"{dtype}-high")
    return ORBCalculator(orbff, device=device)


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int):
    atoms.info["charge"] = charge
    atoms.info["spin"]   = spin


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_orb.csv")
