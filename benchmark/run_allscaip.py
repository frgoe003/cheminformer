#!/usr/bin/env python
"""AllScAIP models via fairchem. Env: ase-uma (fairchem-core >= 2.20.0)"""
from benchmark_common import _ALL_Z, run_benchmark

ALL_MODELS = ["AllScAIP-cons", "AllScAIP-direct"]

_CHECKPOINT = {
    "AllScAIP-cons":   "allscaip-md-conserving-all-omol",
    "AllScAIP-direct": "allscaip-md-direct-all-omol",
}


def supported_elements(model_name: str) -> frozenset:
    return _ALL_Z


def create_calculator(model_name: str, device: str, dtype: str):
    from fairchem.core import FAIRChemCalculator
    fc_device = device.split(":")[0]
    return FAIRChemCalculator.from_model_checkpoint(
        _CHECKPOINT[model_name],
        task_name="omol",
        device=fc_device,
    )


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int):
    atoms.info["charge"] = charge
    atoms.info["spin"]   = spin


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_allscaip.csv")
