#!/usr/bin/env python
"""AceFF-1.1 and AceFF-2.0 models.  Env: ase-aceff"""
from benchmark_common import run_benchmark

ALL_MODELS = ["AceFF-1.1", "AceFF-2.0"]

_CONFIGS = {
    "AceFF-1.1": ("Acellera/AceFF-1.1", "aceff_v1.1.ckpt", None),
    "AceFF-2.0": ("Acellera/AceFF-2.0", "aceff_v2.0.ckpt", 10.0),
}


def supported_elements(model_name: str) -> frozenset:
    from aceff_calculator import ACEFF_ATOMIC_NUMBERS
    return frozenset(ACEFF_ATOMIC_NUMBERS)


def create_calculator(model_name: str, device: str, dtype: str):
    from huggingface_hub import hf_hub_download
    from torchmdnet.calculators import TMDNETCalculator
    repo_id, filename, coulomb_cutoff = _CONFIGS[model_name]
    path   = hf_hub_download(repo_id=repo_id, filename=filename)
    kwargs = {"device": device}
    if coulomb_cutoff is not None:
        kwargs["coulomb_cutoff"] = coulomb_cutoff
    return TMDNETCalculator(path, **kwargs)


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int):
    atoms.info["charge"] = charge
    atoms.info["spin"]   = spin


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_aceff.csv")
