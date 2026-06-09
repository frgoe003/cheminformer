#!/usr/bin/env python
"""MACE models: MACE-POLAR, MACE-OFF23/24, MACE-MH-1, MACE-OMOL-0.  Env: ase-mace"""
from pathlib import Path
from benchmark_common import _ORGANIC_Z, _ALL_Z, run_benchmark, PROJECT

ALL_MODELS = [
    "polar-1-s", "polar-1-m", "polar-1-l",
    "MACE-OFF23(S)", "MACE-OFF23(L)", "MACE-OFF24(M)",
    "MACE-MH-1",
    "MACE-OMOL-0",
]

_POLAR_MODELS    = {"polar-1-s", "polar-1-m", "polar-1-l"}
_MACE_OFF_MODELS = {"MACE-OFF23(S)", "MACE-OFF23(L)", "MACE-OFF24(M)"}
_MACE_MH_MODELS  = {"MACE-MH-1"}
_MACE_OMOL_MODELS = {"MACE-OMOL-0"}

_MACE_OFF_IDS = {
    "MACE-OFF23(S)": "small",
    "MACE-OFF23(L)": "large",
    "MACE-OFF24(M)": "https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true",
}
_MACE_MH_URL = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"


def supported_elements(model_name: str) -> frozenset:
    if model_name in _POLAR_MODELS or model_name in _MACE_OFF_MODELS:
        return _ORGANIC_Z
    return _ALL_Z


def create_calculator(model_name: str, device: str, dtype: str):
    if model_name in _POLAR_MODELS:
        from mace.calculators import mace_polar
        return mace_polar(model=model_name, device=device,
                          default_dtype=dtype, enable_cueq=False)
    if model_name in _MACE_OFF_MODELS:
        from mace.calculators.foundations_models import mace_off
        return mace_off(_MACE_OFF_IDS[model_name], default_dtype=dtype, device=device)
    if model_name in _MACE_MH_MODELS:
        from mace.calculators.foundations_models import mace_mp
        return mace_mp(_MACE_MH_URL, default_dtype=dtype, device=device, head="spice_wB97M")
    if model_name in _MACE_OMOL_MODELS:
        from mace.calculators.foundations_models import mace_omol
        return mace_omol("extra_large", default_dtype=dtype, device=device)
    raise ValueError(f"Unknown model: {model_name!r}")


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int):
    if model_name in _POLAR_MODELS:
        atoms.info["external_field"] = [0.0, 0.0, 0.0]
        atoms.info["charge"] = charge
        atoms.info["spin"]   = spin
    elif model_name in _MACE_OMOL_MODELS:
        atoms.info["charge"] = charge
        atoms.info["spin"]   = spin
    # mace-off*, mace-mh-1: no charge support (matches reference)


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_mace.csv")
