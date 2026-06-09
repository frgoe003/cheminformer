#!/usr/bin/env python
"""OpenMM classical force fields: GAFF2 (small organic) and Amber14 (biomolecules). Env: ase-openmm"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from benchmark_common import _ORGANIC_Z, run_benchmark

# kJ/mol → eV  and  kJ/mol/nm → eV/Å
_KJ_TO_EV        = 1.0 / 96.485332
_KJ_NM_TO_EV_ANG = 1.0 / 96.485332 / 10.0

# H C N O Na Mg P S Cl K Ca Zn — covers standard amino-acid + ion + water elements
_AMBER14_Z = frozenset({1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 30})

ALL_MODELS = ["OpenMM-GAFF2", "OpenMM-Amber14"]


def supported_elements(model_name: str) -> frozenset:
    if model_name == "OpenMM-GAFF2":
        return _ORGANIC_Z
    return _AMBER14_Z


def _omm_platform(device: str):
    from openmm import Platform
    if device.startswith("cuda"):
        idx = device.split(":")[-1] if ":" in device else "0"
        return Platform.getPlatformByName("CUDA"), {"DeviceIndex": idx, "Precision": "mixed"}
    return Platform.getPlatformByName("CPU"), {}


class GAFF2Calculator(Calculator):
    """GAFF2 force field via OpenMM.

    SPICE stores atom-mapped SMILES with explicit H (canonical_order_atoms + mapped=True).
    Molecule.from_mapped_smiles recovers atoms in map-number order, which matches the
    HDF5 conformation array order exactly — no permutation needed.

    Context is cached per SMILES so all conformers of the same molecule skip
    re-parameterisation. Speed-test systems (proteins/water, no SMILES) produce
    status=error — expected behaviour for a small-molecule FF.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, device="cuda:0", **kwargs):
        super().__init__(**kwargs)
        self._device = device
        self._cache: dict = {}  # smiles -> openmm.Context

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        import openmm.unit as unit

        smiles = atoms.info.get("smiles")
        if smiles is None:
            raise RuntimeError(
                "GAFF2 requires atoms.info['smiles'] (SPICE molecules only; "
                "protein speed-test systems are not supported by this FF)."
            )

        context = self._get_or_build(smiles, atoms)
        # Atoms are in HDF5 order in both ASE and OpenMM → set positions directly
        context.setPositions(atoms.get_positions() * 0.1 * unit.nanometer)

        state = context.getState(getEnergy=True, getForces=True)
        self.results["energy"] = (
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * _KJ_TO_EV)
        self.results["forces"] = (
            np.array(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer)) * _KJ_NM_TO_EV_ANG)

    def _get_or_build(self, smiles, atoms):
        if smiles not in self._cache:
            self._cache[smiles] = self._build(smiles, atoms)
        return self._cache[smiles]

    def _build(self, smiles, atoms):
        from openff.toolkit import Molecule
        from openmmforcefields.generators import GAFFTemplateGenerator
        from openmm import app, LangevinMiddleIntegrator, Context
        import openmm.unit as unit

        # SPICE stores atom-mapped SMILES with explicit H (mapped=True,
        # explicit_hydrogens=True) after canonical_order_atoms(). from_mapped_smiles
        # recovers atoms in map-number order, which equals the HDF5 conformation order.
        offmol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)

        if offmol.n_atoms != len(atoms):
            raise ValueError(
                f"SMILES has {offmol.n_atoms} atoms but conformer has {len(atoms)}"
            )

        generator = GAFFTemplateGenerator(molecules=[offmol], forcefield="gaff-2.11")
        ff = app.ForceField()
        ff.registerTemplateGenerator(generator.generator)
        omm_top = offmol.to_topology().to_openmm()
        system = ff.createSystem(omm_top, nonbondedMethod=app.NoCutoff)

        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds)
        platform, props = _omm_platform(self._device)
        return Context(system, integrator, platform, props)


class Amber14Calculator(Calculator):
    """Amber14 (ff14SB + TIP3P) via OpenMM. Requires setup_from_pdb() before calculate().
    configure_atoms() rebuilds ASE Atoms in OpenMM topology order (may add missing H)."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, device="cuda:0", **kwargs):
        super().__init__(**kwargs)
        self._device = device
        self._context = None

    def setup_from_pdb(self, pdb_path: str):
        """Read PDB, optionally add hydrogens, build OpenMM system on GPU.
        Returns (symbols, positions_ang) in OpenMM topology atom order."""
        from openmm.app import PDBFile, ForceField, Modeller, NoCutoff
        from openmm import LangevinMiddleIntegrator, Context
        import openmm.unit as unit

        pdb = PDBFile(pdb_path)
        res_names = {r.name for r in pdb.topology.residues()}
        _WATER_RES = {"HOH", "WAT", "TIP3", "TP3", "SOL"}
        is_water = res_names.issubset(_WATER_RES)
        ff = ForceField("amber14/tip3pfb.xml" if is_water else "amber14-all.xml")

        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)

        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds)
        platform, props = _omm_platform(self._device)
        self._context = Context(system, integrator, platform, props)

        symbols = [a.element.symbol if a.element else "X"
                   for a in modeller.topology.atoms()]
        pos_ang = np.array(modeller.positions.value_in_unit(unit.nanometer)) * 10.0
        return symbols, pos_ang

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if self._context is None:
            raise RuntimeError("Amber14 requires a PDB file — configure_atoms was not called")

        import openmm.unit as unit
        self._context.setPositions(atoms.get_positions() * 0.1 * unit.nanometer)
        state = self._context.getState(getEnergy=True, getForces=True)
        energy = (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                  * _KJ_TO_EV)
        forces = (np.array(state.getForces(asNumpy=True).value_in_unit(
                      unit.kilojoule_per_mole / unit.nanometer))
                  * _KJ_NM_TO_EV_ANG)
        self.results["energy"] = energy
        self.results["forces"] = forces


def configure_atoms(atoms, calc, model_name: str, charge: int, spin: int):
    if model_name != "OpenMM-Amber14":
        return
    pdb_path = atoms.info.get("_pdb_path")
    if not pdb_path:
        return

    symbols, positions = calc.setup_from_pdb(pdb_path)

    # Rebuild ASE Atoms arrays in-place to match OpenMM topology order.
    # addHydrogens may have added H interleaved with heavy atoms, changing both
    # atom count and ordering relative to the original PDB.
    from ase.data import atomic_numbers as _ase_z
    new_numbers = np.array([_ase_z.get(s, 0) for s in symbols], dtype=int)
    atoms.arrays["numbers"]   = new_numbers
    atoms.arrays["positions"] = positions.astype(float)
    # Drop any stale per-atom arrays (e.g. initial_charges) whose length no longer matches
    stale = [k for k, v in atoms.arrays.items()
             if k not in ("numbers", "positions")
             and hasattr(v, "__len__") and len(v) != len(new_numbers)]
    for k in stale:
        del atoms.arrays[k]


def create_calculator(model_name: str, device: str, dtype: str):
    if model_name == "OpenMM-GAFF2":
        return GAFF2Calculator(device=device)
    if model_name == "OpenMM-Amber14":
        return Amber14Calculator(device=device)
    raise ValueError(f"Unknown model: {model_name!r}")


if __name__ == "__main__":
    run_benchmark(ALL_MODELS, supported_elements, "results_openmm.csv")
