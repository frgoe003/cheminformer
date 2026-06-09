#!/usr/bin/env python
"""UMA-s-1 and UMA-m-1 models.  Env: ase-uma"""
from benchmark_common import _ALL_Z, run_benchmark, PROJECT

ALL_MODELS = ["UMA-s-1", "UMA-m-1"]

_FILES = {
    "UMA-s-1": PROJECT / "uma-s-1p2.pt",
    "UMA-m-1": PROJECT / "uma-m-1p1.pt",
}


def supported_elements(model_name: str) -> frozenset:
    return _ALL_Z


def create_calculator(model_name: str, device: str, dtype: str):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    fc_device = device.split(":")[0]
    predictor = load_predict_unit(path=_FILES[model_name], device=fc_device,
                                  inference_settings="default")
    return FAIRChemCalculator(predictor, task_name="omol")


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int):
    atoms.info["charge"] = charge
    atoms.info["spin"]   = spin


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_uma.csv")
