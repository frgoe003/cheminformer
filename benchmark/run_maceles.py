#!/usr/bin/env python
"""MACELES-OFF model.  Env: ase-maceles"""
from benchmark_common import _ORGANIC_Z, run_benchmark

ALL_MODELS = ["MACELES-OFF"]

_URL = "https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/MACELES-OFF_small_converted.model?raw=true"


def supported_elements(model_name: str) -> frozenset:
    return _ORGANIC_Z


def create_calculator(model_name: str, device: str, dtype: str):
    from mace.calculators.foundations_models import mace_off
    return mace_off(_URL, default_dtype=dtype, device=device)


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_maceles.csv")
