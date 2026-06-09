#!/usr/bin/env python
"""Egret-1 model.  Env: ase-egret"""
from benchmark_common import _ORGANIC_Z, run_benchmark

ALL_MODELS = ["Egret-1"]

_URL = "https://github.com/rowansci/egret-public/blob/master/compiled_models/EGRET_1.model?raw=true"


def supported_elements(model_name: str) -> frozenset:
    return _ORGANIC_Z


def create_calculator(model_name: str, device: str, dtype: str):
    from mace.calculators.foundations_models import mace_off
    return mace_off(_URL, default_dtype=dtype, device=device)


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_egret.csv")
